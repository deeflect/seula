import "./site.css";
import "./landing.css";

// Copy pill: the check shows for 1.5 s, the text stays, a polite status announces it.
const status = document.getElementById("copy-status")!;
for (const b of document.querySelectorAll<HTMLButtonElement>("[data-copy]")) {
  const [copyIcon, checkIcon] = b.querySelectorAll("svg");
  let t = 0;
  b.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(b.dataset.copy!); } catch { return; }
    copyIcon.toggleAttribute("hidden", true); checkIcon.toggleAttribute("hidden", false);
    status.textContent = "Copied";
    clearTimeout(t);
    t = window.setTimeout(() => { copyIcon.toggleAttribute("hidden", false); checkIcon.toggleAttribute("hidden", true); status.textContent = ""; }, 1500);
  });
}
