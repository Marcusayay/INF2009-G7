import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite' // Import the new plugin

export default defineConfig({
  // Add the tailwindcss plugin here
  plugins: [react(), tailwindcss()],
})