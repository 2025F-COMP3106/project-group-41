# Minimal React Frontend

This is a simple React UI to upload image files and show prediction output from an API.

## Run

From project root:

```bash
python -m http.server 5173
```

Then open:

`http://127.0.0.1:5173/frontend/`

## Expected API

- Method: `POST`
- URL: set in the "Prediction API URL" input (default `http://127.0.0.1:8000/predict`)
- Request body: `multipart/form-data` with field name `file`
- Response JSON example:

```json
{
  "prediction": "BENIGN",
  "confidence": 0.91,
  "adjusted_confidence": 0.88
}
```

If your backend returns plain text, it will still be displayed in the result card.
