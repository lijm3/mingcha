import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 开发模式把 /api 与 /artifacts 代理到后端（§4.3），生产由后端托管 dist、同源无需代理。
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://localhost:8000', changeOrigin: true },
      '/artifacts': { target: 'http://localhost:8000', changeOrigin: true },
    },
  },
  build: { outDir: 'dist' },
})
