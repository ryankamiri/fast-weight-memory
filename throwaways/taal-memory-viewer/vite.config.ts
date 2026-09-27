import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath } from 'node:url';
import { localTraces } from './server/local-traces.ts';

export default defineConfig({
  plugins: [react(), localTraces(fileURLToPath(new URL('./data', import.meta.url)))],
});
