document.addEventListener("DOMContentLoaded", () => {
    // Elements
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("fileInput");
    const dropzonePrompt = document.getElementById("dropzonePrompt");
    const previewArea = document.getElementById("previewArea");
    const imagePreview = document.getElementById("imagePreview");
    const fileName = document.getElementById("fileName");
    const fileSize = document.getElementById("fileSize");
    const btnRemove = document.getElementById("btnRemove");
    const btnClassify = document.getElementById("btnClassify");
    const loadingState = document.getElementById("loadingState");
    const errorBanner = document.getElementById("errorBanner");
    const errorMessage = document.getElementById("errorMessage");
    const resultsSection = document.getElementById("resultsSection");
    const engineBadge = document.getElementById("engineBadge");
    const inferenceTime = document.getElementById("inferenceTime");
    const statusBadge = document.getElementById("statusBadge");
    const statusText = document.getElementById("statusText");

    // Baseline elements
    const baselineTopClass = document.getElementById("baselineTopClass");
    const baselineConfidence = document.getElementById("baselineConfidence");
    const baselineBreakdown = document.getElementById("baselineBreakdown");

    // Optimized elements
    const optimizedTopClass = document.getElementById("optimizedTopClass");
    const optimizedConfidence = document.getElementById("optimizedConfidence");
    const optimizedBreakdown = document.getElementById("optimizedBreakdown");

    let selectedFile = null;

    // 1. Health check
    async function checkHealth() {
        try {
            const res = await fetch("/api/health");
            const data = await res.json();
            if (data.status === "online") {
                statusBadge.className = "status-badge online";
                statusText.textContent = `🟢 Engine Ready (${data.mode})`;
            } else {
                statusBadge.className = "status-badge offline";
                statusText.textContent = "🔴 Engine Disconnected";
            }
        } catch (e) {
            statusBadge.className = "status-badge offline";
            statusText.textContent = "🔴 Status Check Failed";
        }
    }
    checkHealth();
    setInterval(checkHealth, 10000);

    // 2. File Selection & Drag-and-Drop
    dropzone.addEventListener("click", (e) => {
        if (e.target !== btnRemove && !previewArea.contains(e.target)) {
            fileInput.click();
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0]);
        }
    });

    ["dragenter", "dragover"].forEach(evt => {
        dropzone.addEventListener(evt, (e) => {
            e.preventDefault();
            dropzone.classList.add("dragover");
        });
    });

    ["dragleave", "drop"].forEach(evt => {
        dropzone.addEventListener(evt, (e) => {
            e.preventDefault();
            dropzone.classList.remove("dragover");
        });
    });

    dropzone.addEventListener("drop", (e) => {
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith("image/")) {
            showError("Please select an image file (PNG, JPG, WEBP).");
            return;
        }

        hideError();
        selectedFile = file;
        fileName.textContent = file.name;
        fileSize.textContent = `${(file.size / 1024).toFixed(1)} KB`;

        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropzonePrompt.classList.add("hidden");
            previewArea.classList.remove("hidden");
            btnClassify.disabled = false;
            resultsSection.classList.add("hidden");
        };
        reader.readAsDataURL(file);
    }

    btnRemove.addEventListener("click", (e) => {
        e.stopPropagation();
        selectedFile = null;
        fileInput.value = "";
        imagePreview.src = "";
        dropzonePrompt.classList.remove("hidden");
        previewArea.classList.add("hidden");
        btnClassify.disabled = true;
        resultsSection.classList.add("hidden");
        hideError();
    });

    // 3. Classify Request
    btnClassify.addEventListener("click", async () => {
        if (!selectedFile) return;

        hideError();
        resultsSection.classList.add("hidden");
        loadingState.classList.remove("hidden");
        btnClassify.disabled = true;

        const formData = new FormData();
        formData.append("image", selectedFile);

        try {
            const response = await fetch("/classify", {
                method: "POST",
                body: formData
            });

            const data = await response.json();
            loadingState.classList.add("hidden");
            btnClassify.disabled = false;

            if (!data.success) {
                showError(data.error || "Classification failed.");
                return;
            }

            // Render side-by-side results
            renderComparison(data);
        } catch (err) {
            loadingState.classList.add("hidden");
            btnClassify.disabled = false;
            showError(`Error connecting to server: ${err.message}`);
        }
    });

    // 4. Render Results
    function renderComparison(data) {
        engineBadge.textContent = data.engine;
        inferenceTime.textContent = `⚡ ${data.predict_time}`;

        // Populate Baseline
        baselineTopClass.textContent = data.baseline.top_class;
        baselineConfidence.textContent = `${data.baseline.confidence_pct} Confidence`;
        renderBreakdownList(baselineBreakdown, data.baseline.breakdown);

        // Populate Optimized
        optimizedTopClass.textContent = data.optimized.top_class;
        optimizedConfidence.textContent = `${data.optimized.confidence_pct} Confidence`;
        renderBreakdownList(optimizedBreakdown, data.optimized.breakdown);

        resultsSection.classList.remove("hidden");
    }

    function renderBreakdownList(container, items) {
        container.innerHTML = "";
        items.forEach((item, index) => {
            const row = document.createElement("div");
            row.className = "breakdown-item";
            row.innerHTML = `
                <div class="item-info">
                    <span>#${index + 1} ${item.class}</span>
                    <span>${item.percentage}</span>
                </div>
                <div class="progress-track">
                    <div class="progress-fill" style="width: ${Math.max(item.percent_num, 3)}%"></div>
                </div>
            `;
            container.appendChild(row);
        });
    }

    function showError(msg) {
        errorMessage.textContent = msg;
        errorBanner.classList.remove("hidden");
    }

    function hideError() {
        errorBanner.classList.add("hidden");
    }
});
