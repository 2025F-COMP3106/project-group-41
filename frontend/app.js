function App() {
  const { useEffect, useMemo, useState } = React;
  const [apiUrl, setApiUrl] = useState("http://127.0.0.1:8000/predict");
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [results, setResults] = useState([]);

  const canSubmit = useMemo(() => apiUrl.trim() && files.length > 0 && !loading, [apiUrl, files, loading]);

  const onFileChange = (event) => {
    setError("");
    setResults([]);
    setFiles(Array.from(event.target.files || []));
  };

  const getImageFilesFromClipboard = (event) => {
    const clipboardItems = event.clipboardData?.items || [];
    const pastedFiles = [];
    for (const item of clipboardItems) {
      if (item.type && item.type.startsWith("image/")) {
        const file = item.getAsFile();
        if (file) {
          pastedFiles.push(file);
        }
      }
    }
    return pastedFiles;
  };

  const onPasteImages = (event) => {
    const pastedFiles = getImageFilesFromClipboard(event);
    if (pastedFiles.length === 0) {
      return;
    }
    event.preventDefault();
    setError("");
    setResults([]);
    setFiles((prevFiles) => [...prevFiles, ...pastedFiles]);
  };

  useEffect(() => {
    const handleWindowPaste = (event) => onPasteImages(event);
    window.addEventListener("paste", handleWindowPaste);
    return () => window.removeEventListener("paste", handleWindowPaste);
  }, []);

  const parseServerResponse = async (response) => {
    const contentType = response.headers.get("content-type") || "";
    if (contentType.includes("application/json")) {
      return await response.json();
    }
    return { raw: await response.text() };
  };

  const handleUpload = async (event) => {
    event.preventDefault();
    if (!canSubmit) return;

    setLoading(true);
    setError("");
    setResults([]);

    try {
      const collected = [];
      for (const file of files) {
        const formData = new FormData();
        formData.append("file", file);

        const response = await fetch(apiUrl, {
          method: "POST",
          body: formData
        });

        const payload = await parseServerResponse(response);
        if (!response.ok) {
          throw new Error(payload?.error || payload?.message || payload?.detail || `Request failed (${response.status})`);
        }

        collected.push({
          filename: file.name,
          ...payload
        });
      }
      setResults(collected);
    } catch (uploadError) {
      if (uploadError.message === "Failed to fetch") {
        setError("Could not connect to the server. Make sure the backend is running: python server.py");
      } else {
        setError(uploadError.message || "Something went wrong.");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="container">
      <img className="hero-image" src="../Cancer-Tron-300.png" alt="Cancer-Tron 3000" />
      <h1>Skin Lesion Result Checker</h1>
      <p className="subtitle">Upload one or more images, then get prediction results.</p>
      <section className="instructions">
        <h2>How to use</h2>
        <ol>
          <li>Train the model (if not done yet): <code>python codebase/main.py</code></li>
          <li>Start the backend server: <code>python server.py</code></li>
          <li>Upload image(s) using the file picker, or copy an image and press <strong>Ctrl+V</strong> anywhere on this page.</li>
          <li>Click <strong>Upload and Get Result</strong> to get the prediction.</li>
        </ol>
      </section>

      <form onSubmit={handleUpload} className="card">
        <label htmlFor="api-url">Prediction API URL</label>
        <input
          id="api-url"
          type="url"
          value={apiUrl}
          onChange={(e) => setApiUrl(e.target.value)}
          placeholder="http://127.0.0.1:8000/predict"
          required
        />

        <label htmlFor="images">Choose image(s)</label>
        <input
          id="images"
          type="file"
          accept="image/*"
          multiple
          onChange={onFileChange}
        />
        <div className="paste-box" tabIndex="0" onPaste={onPasteImages}>
          Paste zone: click here and press <strong>Ctrl+V</strong> to add copied image(s).
        </div>
        {files.length > 0 && (
          <p className="selected-files">Selected images: {files.length}</p>
        )}

        <button type="submit" disabled={!canSubmit}>
          {loading ? "Uploading..." : "Upload and Get Result"}
        </button>
      </form>

      {error && <p className="error">{error}</p>}

      {results.length > 0 && (
        <section className="results">
          <h2>Results</h2>
          {results.map((item, index) => (
            <article className="result-card" key={`${item.filename}-${index}`}>
              <h3>{item.filename}</h3>
              <p><strong>Prediction:</strong> {item.prediction ?? item.label ?? "N/A"}</p>
              <p><strong>Confidence:</strong> {item.confidence ? `${(item.confidence * 100).toFixed(2)}%` : "N/A"}</p>
              {"adjusted_confidence" in item && (
                <p><strong>Adjusted confidence:</strong> {(item.adjusted_confidence * 100).toFixed(2)}%</p>
              )}
              {"flag_for_review" in item && (
                <p className={item.flag_for_review ? "review-flag" : "review-ok"}>
                  {item.flag_for_review
                    ? "Low confidence — flagged for dermatologist review"
                    : "High confidence — model decision accepted"}
                </p>
              )}
              {item.error && <p className="error">{item.error}</p>}
              {item.raw && <pre>{item.raw}</pre>}
            </article>
          ))}
        </section>
      )}
    </main>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
