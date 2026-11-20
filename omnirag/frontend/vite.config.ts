import path from 'node:path'

import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(import.meta.dirname, './src'),
    },
  },
  server: {
    port: 5173,
    // The backend (FastAPI/uvicorn) runs on 8000 by default — see
    // backend/README.md. Proxying /api means the frontend never needs a
    // hardcoded absolute backend URL in dev, and CORS is a non-issue
    // locally since the browser only ever talks to :5173.
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
