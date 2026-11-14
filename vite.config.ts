import { resolve } from "node:path";
import { type Plugin, defineConfig } from "vite";

/** Serve /app as app.html in dev and preview, matching Vercel's cleanUrls. */
function cleanUrls(): Plugin {
  const rewrite = (req: { url?: string }, _res: unknown, next: () => void) => {
    if (req.url === "/app" || req.url?.startsWith("/app?")) req.url = req.url.replace("/app", "/app.html");
    next();
  };
  return {
    name: "clean-urls",
    configureServer: (s) => void s.middlewares.use(rewrite),
    configurePreviewServer: (s) => void s.middlewares.use(rewrite),
  };
}

export default defineConfig({
  plugins: [cleanUrls()],
  build: {
    target: "es2022",
    rollupOptions: {
      input: {
        index: resolve(import.meta.dirname, "index.html"),
        app: resolve(import.meta.dirname, "app.html"),
      },
    },
  },
});
