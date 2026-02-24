/**
 * ELA Guard — Main JavaScript
 * Handles drag & drop upload, preview, and form submission.
 */

(function () {
  const dropZone   = document.getElementById("drop-zone");
  const dropInner  = document.getElementById("drop-inner");
  const previewWrap= document.getElementById("preview-wrap");
  const previewImg = document.getElementById("preview-img");
  const previewName= document.getElementById("preview-name");
  const previewSize= document.getElementById("preview-size");
  const imageInput = document.getElementById("image-input");
  const analyzeBtn = document.getElementById("analyze-btn");
  const clearBtn   = document.getElementById("clear-btn");
  const uploadForm = document.getElementById("upload-form");
  const btnText    = document.querySelector(".btn-text");
  const spinner    = document.getElementById("spinner");

  if (!dropZone) return;  // Not on upload page

  const MAX_SIZE = 10 * 1024 * 1024;
  const ALLOWED_TYPES = new Set(["image/jpeg", "image/png", "image/webp", "image/gif", "image/heic"]);

  // Click to open file dialog
  dropZone.addEventListener("click", (e) => {
    if (e.target === clearBtn || clearBtn?.contains(e.target)) return;
    imageInput.click();
  });

  // File selected via input
  imageInput.addEventListener("change", () => {
    if (imageInput.files[0]) loadPreview(imageInput.files[0]);
  });

  // Drag & Drop
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file) {
      // Inject into input
      const dt = new DataTransfer();
      dt.items.add(file);
      imageInput.files = dt.files;
      loadPreview(file);
    }
  });

  // Clear button
  clearBtn?.addEventListener("click", (e) => {
    e.stopPropagation();
    clearPreview();
  });

  function loadPreview(file) {
    if (!ALLOWED_TYPES.has(file.type)) {
      showError("Unsupported file type. Please use PNG, JPG, JPEG, WebP, GIF, or HEIC.");
      return;
    }
    if (file.size > MAX_SIZE) {
      showError("File is too large. Maximum allowed size is 10 MB.");
      return;
    }
    const reader = new FileReader();
    reader.onload = (ev) => {
      previewImg.src = ev.target.result;
      previewName.textContent = file.name;
      previewSize.textContent = formatBytes(file.size);
      dropInner.classList.add("hidden");
      previewWrap.classList.remove("hidden");
      analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  function clearPreview() {
    imageInput.value = "";
    previewImg.src = "";
    previewWrap.classList.add("hidden");
    dropInner.classList.remove("hidden");
    analyzeBtn.disabled = true;
  }

  function showError(msg) {
    clearPreview();
    // Re-use the alert area if present, else alert
    const existingAlert = document.querySelector(".alert-error");
    if (existingAlert) {
      existingAlert.innerHTML = `<span class="alert-icon">⚠</span> ${msg}`;
    } else {
      alert(msg);
    }
  }

  // Show spinner on submit
  uploadForm?.addEventListener("submit", () => {
    if (!analyzeBtn.disabled) {
      btnText.classList.add("hidden");
      spinner.classList.remove("hidden");
      analyzeBtn.disabled = true;
    }
  });

  function formatBytes(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(2) + " MB";
  }
})();
