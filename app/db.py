"""SQLite persistence (parameterised queries only)."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS portfolios(
  id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, base_currency TEXT NOT NULL DEFAULT 'USD',
  cash REAL NOT NULL DEFAULT 0, benchmark TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS holdings(
  id INTEGER PRIMARY KEY AUTOINCREMENT, portfolio_id INTEGER NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
  symbol TEXT NOT NULL, asset_type TEXT NOT NULL, quantity REAL NOT NULL, average_cost REAL NOT NULL DEFAULT 0,
  updated_at TEXT NOT NULL, UNIQUE(portfolio_id, symbol));
CREATE TABLE IF NOT EXISTS transactions(
  id INTEGER PRIMARY KEY AUTOINCREMENT, portfolio_id INTEGER NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
  symbol TEXT NOT NULL, side TEXT NOT NULL, quantity REAL NOT NULL, price REAL NOT NULL, fee REAL NOT NULL DEFAULT 0,
  executed_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS target_allocations(
  id INTEGER PRIMARY KEY AUTOINCREMENT, portfolio_id INTEGER NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
  version INTEGER NOT NULL, symbol TEXT NOT NULL, weight REAL NOT NULL, min_weight REAL, max_weight REAL,
  effective_from TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS rebalance_runs(
  id INTEGER PRIMARY KEY AUTOINCREMENT, portfolio_id INTEGER NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
  target_version INTEGER NOT NULL, created_at TEXT NOT NULL, config_json TEXT NOT NULL, market_json TEXT NOT NULL,
  sentiment_json TEXT NOT NULL, summary_json TEXT NOT NULL, status TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS proposed_trades(
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER NOT NULL REFERENCES rebalance_runs(id) ON DELETE CASCADE,
  symbol TEXT NOT NULL, action TEXT NOT NULL, quantity REAL NOT NULL, price REAL NOT NULL, value REAL NOT NULL,
  fee REAL NOT NULL, review_priority TEXT, rationale TEXT);
CREATE TABLE IF NOT EXISTS backtests(
  id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, config_json TEXT NOT NULL, result_json TEXT NOT NULL);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Repository:
    def __init__(self, path: str = ":memory:") -> None:
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.lock = threading.RLock()
        with self.lock:
            self.conn.executescript(SCHEMA)

    def _q(self, sql: str, args: tuple = ()) -> List[Dict[str, Any]]:
        with self.lock:
            return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    def _x(self, sql: str, args: tuple = ()) -> int:
        with self.lock:
            cur = self.conn.execute(sql, args)
            self.conn.commit()
            # INSERT -> new row id; UPDATE/DELETE -> number of affected rows
            return cur.lastrowid if sql.lstrip().upper().startswith("INSERT") else cur.rowcount

    # portfolios
    def create_portfolio(self, name: str, base_currency: str = "USD", cash: float = 0.0, benchmark: Optional[str] = None) -> int:
        return self._x("INSERT INTO portfolios(name,base_currency,cash,benchmark,created_at) VALUES(?,?,?,?,?)",
                       (name, base_currency, cash, benchmark, _now()))

    def list_portfolios(self): return self._q("SELECT * FROM portfolios ORDER BY id")

    def get_portfolio(self, pid: int):
        r = self._q("SELECT * FROM portfolios WHERE id=?", (pid,))
        return r[0] if r else None

    def update_portfolio(self, pid: int, **fields) -> bool:
        allowed = {k: v for k, v in fields.items() if k in ("name", "base_currency", "cash", "benchmark") and v is not None}
        if not allowed:
            return self.get_portfolio(pid) is not None
        sets = ", ".join(f"{k}=?" for k in allowed)  # keys are whitelisted above
        return self._x(f"UPDATE portfolios SET {sets} WHERE id=?", (*allowed.values(), pid)) > 0

    def delete_portfolio(self, pid: int) -> bool:
        return self._x("DELETE FROM portfolios WHERE id=?", (pid,)) > 0

    # holdings
    def upsert_holding(self, pid: int, symbol: str, asset_type: str, quantity: float, average_cost: float = 0.0) -> None:
        self._x("""INSERT INTO holdings(portfolio_id,symbol,asset_type,quantity,average_cost,updated_at)
                   VALUES(?,?,?,?,?,?) ON CONFLICT(portfolio_id,symbol)
                   DO UPDATE SET quantity=excluded.quantity, average_cost=excluded.average_cost,
                                 asset_type=excluded.asset_type, updated_at=excluded.updated_at""",
                (pid, symbol, asset_type, quantity, average_cost, _now()))
        self._x("INSERT INTO transactions(portfolio_id,symbol,side,quantity,price,fee,executed_at) VALUES(?,?,?,?,?,?,?)",
                (pid, symbol, "SET", quantity, average_cost, 0.0, _now()))

    def replace_portfolio(self, pid: int, rows: List[dict]) -> None:
        """Atomically replace a portfolio's cash and holdings from validated CSV rows."""
        cash = next((r["quantity"] for r in rows if r["symbol"] == "CASH"), 0.0)
        now = _now()
        with self.lock:
            try:
                self.conn.execute("UPDATE portfolios SET cash=? WHERE id=?", (cash, pid))
                self.conn.execute("DELETE FROM holdings WHERE portfolio_id=?", (pid,))
                self.conn.executemany(
                    "INSERT INTO holdings(portfolio_id,symbol,asset_type,quantity,average_cost,updated_at) VALUES(?,?,?,?,?,?)",
                    [(pid, r["symbol"], r["asset_type"], r["quantity"], r["average_cost"], now)
                     for r in rows if r["symbol"] != "CASH"],
                )
                self.conn.executemany(
                    "INSERT INTO transactions(portfolio_id,symbol,side,quantity,price,fee,executed_at) VALUES(?,?,?,?,?,?,?)",
                    [(pid, r["symbol"], "SET", r["quantity"], r["average_cost"], 0.0, now)
                     for r in rows if r["symbol"] != "CASH"],
                )
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    def list_holdings(self, pid: int): return self._q("SELECT * FROM holdings WHERE portfolio_id=? ORDER BY symbol", (pid,))
    def delete_holding(self, pid: int, symbol: str) -> bool:
        return self._x("DELETE FROM holdings WHERE portfolio_id=? AND symbol=?", (pid, symbol)) > 0
    def list_transactions(self, pid: int): return self._q("SELECT * FROM transactions WHERE portfolio_id=? ORDER BY id", (pid,))

    # targets (versioned)
    def set_targets(self, pid: int, targets: Dict[str, float], ranges: Optional[Dict[str, tuple]] = None) -> int:
        ranges = ranges or {}
        with self.lock:
            v = (self.conn.execute("SELECT COALESCE(MAX(version),0) FROM target_allocations WHERE portfolio_id=?", (pid,)).fetchone()[0]) + 1
            now = _now()
            for s, w in targets.items():
                lo, hi = ranges.get(s, (None, None))
                self.conn.execute("INSERT INTO target_allocations(portfolio_id,version,symbol,weight,min_weight,max_weight,effective_from) VALUES(?,?,?,?,?,?,?)",
                                  (pid, v, s, w, lo, hi, now))
            self.conn.commit()
        return v

    def get_targets(self, pid: int):
        rows = self._q("""SELECT * FROM target_allocations WHERE portfolio_id=? AND version=
                          (SELECT MAX(version) FROM target_allocations WHERE portfolio_id=?)""", (pid, pid))
        return (rows[0]["version"] if rows else 0), rows

    # runs
    def save_run(self, pid: int, version: int, config: dict, market: dict, sentiment: dict, summary: dict, trades: List[dict], status: str) -> int:
        rid = self._x("INSERT INTO rebalance_runs(portfolio_id,target_version,created_at,config_json,market_json,sentiment_json,summary_json,status) VALUES(?,?,?,?,?,?,?,?)",
                      (pid, version, _now(), json.dumps(config), json.dumps(market), json.dumps(sentiment), json.dumps(summary), status))
        for t in trades:
            self._x("INSERT INTO proposed_trades(run_id,symbol,action,quantity,price,value,fee,review_priority,rationale) VALUES(?,?,?,?,?,?,?,?,?)",
                    (rid, t["symbol"], t["action"], t["quantity"], t["price"], t["value"], t["fee"], t.get("review_priority"), t.get("rationale")))
        return rid

    def get_run(self, rid: int):
        r = self._q("SELECT * FROM rebalance_runs WHERE id=?", (rid,))
        if not r:
            return None
        run = r[0]
        run["trades"] = self._q("SELECT * FROM proposed_trades WHERE run_id=? ORDER BY id", (rid,))
        for k in ("config_json", "market_json", "sentiment_json", "summary_json"):
            run[k[:-5]] = json.loads(run.pop(k))
        return run

    def save_backtest(self, config: dict, result: dict) -> int:
        return self._x("INSERT INTO backtests(created_at,config_json,result_json) VALUES(?,?,?)", (_now(), json.dumps(config), json.dumps(result)))

    def get_backtest(self, bid: int):
        r = self._q("SELECT * FROM backtests WHERE id=?", (bid,))
        if not r:
            return None
        return {"id": r[0]["id"], "created_at": r[0]["created_at"], "config": json.loads(r[0]["config_json"]), "result": json.loads(r[0]["result_json"])}
