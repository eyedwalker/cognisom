import { defineConfig } from 'vite';

export default defineConfig({
  base: '/rtx-viewer/',
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
  },
  server: {
    port: 3000,
  },
});
