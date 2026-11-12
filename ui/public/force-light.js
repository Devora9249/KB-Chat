// Light mode only. Chainlit reads the theme from localStorage["vite-ui-theme"] and only
// falls back to `default_theme` when it is empty, so a browser that once chose "dark"
// (or "system" on a dark OS) would stay dark. This runs before Chainlit loads and resets it.
try {
  localStorage.setItem("vite-ui-theme", "light");
} catch {
  // localStorage blocked (e.g. strict privacy mode): Chainlit then uses default_theme = "light".
}
