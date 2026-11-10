import "./landing.css";

// Copy buttons: the check shows for 1.5 s, a polite status announces it. data-copy is read at click time (the code card changes it).
const status = document.getElementById("copy-status")!;
document.addEventListener("click", async (e) => {
  const b = (e.target as HTMLElement).closest<HTMLButtonElement>("[data-copy]");
  if (!b) return;
  try { await navigator.clipboard.writeText(b.dataset.copy!); } catch { return; }
  const [copyIcon, checkIcon] = b.querySelectorAll("svg");
  copyIcon.toggleAttribute("hidden", true); checkIcon.toggleAttribute("hidden", false);
  status.textContent = "Copied";
  clearTimeout(+(b.dataset.t ?? 0));
  b.dataset.t = String(window.setTimeout(() => { copyIcon.toggleAttribute("hidden", false); checkIcon.toggleAttribute("hidden", true); status.textContent = ""; }, 1500));
});

// "What it reads" draws its glosses in when it scrolls into view (only if it starts below the fold).
const reads = document.querySelector<HTMLElement>(".reads")!;
reads.querySelectorAll("li").forEach((li, i) => li.style.setProperty("--i", String(i)));
if (reads.getBoundingClientRect().top > innerHeight && "IntersectionObserver" in window) {
  reads.classList.add("pending");
  const io = new IntersectionObserver(([e]) => { if (e.isIntersecting) { reads.classList.add("draw"); io.disconnect(); } }, { threshold: 0.3 });
  io.observe(reads);
}

// The model and its weights load after first paint: when the browser is idle, or at the first interaction.
let started = false;
const start = () => { if (!started) { started = true; import("./landing-live"); } };
for (const t of ["pointerdown", "keydown", "focusin"]) addEventListener(t, start, { once: true, passive: true });
addEventListener("load", () => ("requestIdleCallback" in window ? requestIdleCallback(start, { timeout: 2000 }) : setTimeout(start, 300)));
