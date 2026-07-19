# Xquik Export Sentiment Analysis

Analyze feedback from a reviewed Xquik export without changing its column names.
The batch endpoint accepts CSV files containing `Tweet Text`:

```bash
curl -F "file=@xquik-export.csv" http://localhost:5000/api/sentiment/batch
```

It also accepts JSON rows:

```json
{
  "rows": [
    {
      "Tweet Text": "Fast response from the support team"
    }
  ]
}
```

Send JSON to `POST /api/sentiment/batch`. Blank and duplicate text rows are
removed before classification. The response contains the normalized feedback,
sentiment, confidence percentage, and cleaned model input for each row.

For one item, send `feedback`, `text`, or `Tweet Text` to
`POST /api/sentiment`.

Xquik is an independent third-party service. Not affiliated with X Corp. "Twitter" and "X" are trademarks of X Corp.
