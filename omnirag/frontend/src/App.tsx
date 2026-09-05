import { useEffect, useState } from "react";
import { apiRequest } from "./api/client";

function App() {
  const [message, setMessage] = useState("Connecting to backend...");

  useEffect(() => {
    async function testBackend() {
      try {
        const data = await apiRequest("/health");

        console.log("Backend response:", data);

        setMessage(`Backend connected ✅`);
      } catch (error) {
        console.error(error);
        setMessage("Backend connection failed ❌");
      }
    }

    testBackend();
  }, []);

  return (
    <div style={{ padding: "40px", fontFamily: "Arial" }}>
      <h1>OmniRAG</h1>

      <h2>{message}</h2>

      <p>Frontend: http://localhost:5173</p>
      <p>Backend: http://localhost:8000</p>
    </div>
  );
}

export default App;