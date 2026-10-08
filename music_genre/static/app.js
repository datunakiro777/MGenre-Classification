(() => {
  const form = document.getElementById("analysis-form");
  const fileInput = document.getElementById("audio-input");
  const urlInput = document.getElementById("media-url");
  const dropZone = document.getElementById("drop-zone");
  const fileLabel = document.getElementById("file-label");

  function updateFile() {
    const file = fileInput.files?.[0];
    fileLabel.textContent = file ? file.name : "Drop a file or tap to browse";
    if (file) urlInput.value = "";
  }

  fileInput.addEventListener("change", updateFile);
  urlInput.addEventListener("input", () => {
    if (urlInput.value.trim() && fileInput.files?.length) {
      fileInput.value = "";
      updateFile();
    }
  });

  for (const eventName of ["dragenter", "dragover"]) {
    dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropZone.classList.add("is-dragging");
    });
  }
  for (const eventName of ["dragleave", "drop"]) {
    dropZone.addEventListener(eventName, () => dropZone.classList.remove("is-dragging"));
  }
  dropZone.addEventListener("drop", (event) => {
    event.preventDefault();
    const file = event.dataTransfer?.files?.[0];
    if (!file) return;
    const transfer = new DataTransfer();
    transfer.items.add(file);
    fileInput.files = transfer.files;
    updateFile();
  });

  form.addEventListener("submit", (event) => {
    if (fileInput.files?.length || urlInput.value.trim()) return;
    event.preventDefault();
    urlInput.setCustomValidity("Choose a file or paste a direct media link.");
    urlInput.reportValidity();
    urlInput.setCustomValidity("");
  });

  if (window.matchMedia("(max-width: 760px)").matches) {
    const destination = document.querySelector('[role="alert"]') ||
      (document.body.dataset.hasResult === "true" && document.getElementById("result"));
    if (destination) requestAnimationFrame(() => destination.scrollIntoView({ block: "start" }));
  }
})();
