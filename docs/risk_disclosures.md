# Risk disclosures and known limitations

- **Not investment advice.** The software proposes trades for manual review. It never places orders.
- **Sentiment is an uncertain signal.** It is not a price forecast. The tilt formula `w*exp(k*S)` is an experimental heuristic.
- **Demo data is synthetic.** Default prices are generated random walks; default news is generated headlines whose sentiment is
  independent of prices *by construction*. Any backtest on them tests the software, not markets, and cannot show that sentiment helps.
- **Live providers are untested here.** Yahoo/Alpha Vantage clients exist and their response *parsers* are unit-tested on mocked payloads,
  but the HTTP calls were not exercised against the live services. Check each provider's terms before use (Yahoo's endpoint is unofficial).
- **FinBERT is optional and unexercised in the test suite** (needs internet + torch). The default classifier is a rule-based
  lexicon model. Its 95% accuracy on `data/labeled_headlines.csv` is measured on 40 headlines written by the same author as the lexicon:
  it is a sanity check, not a benchmark. Validate any model on real, independently labelled headlines.
- **API has no authentication** (single-user MVP). Do not expose it publicly as-is.
- **Backtest simplifications:** daily closes, flat percentage fee, no slippage/spread/tax/liquidity model, 0% risk-free rate,
  1-day execution lag, no survivorship-bias handling. Past results do not predict future results.
- **Single-currency only;** no FX conversion. Crypto is not specially handled (24/7 calendars are not modelled).
