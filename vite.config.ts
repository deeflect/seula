import { defineConfig } from "vite";

// Two pages: the landing at / and the live demo at /demo (demo.html also serves ?lab and ?invoice).
export default defineConfig({
  build: { rollupOptions: { input: { main: "index.html", demo: "demo.html" } } },
});
