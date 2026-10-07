const questionInput = document.getElementById("question");
const askButton = document.getElementById("askButton");
const buttonText = document.getElementById("buttonText");

const loading = document.getElementById("loading");
const result = document.getElementById("result");
const error = document.getElementById("error");

const answerElement = document.getElementById("answer");
const sourcesElement = document.getElementById("sources");
const errorMessage = document.getElementById("errorMessage");


// --------------------------------------------------
// Upload elements
// --------------------------------------------------

const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const chooseFileButton = document.getElementById("chooseFileButton");

const selectedFile = document.getElementById("selectedFile");
const fileName = document.getElementById("fileName");
const fileSize = document.getElementById("fileSize");
const removeFileButton = document.getElementById("removeFileButton");

const uploadButton = document.getElementById("uploadButton");
const uploadButtonText = document.getElementById("uploadButtonText");

const uploadStatus = document.getElementById("uploadStatus");
const uploadSpinner = document.getElementById("uploadSpinner");
const uploadStatusTitle = document.getElementById("uploadStatusTitle");
const uploadStatusText = document.getElementById("uploadStatusText");

const activePaper = document.getElementById("activePaper");
const activePaperName = document.getElementById("activePaperName");
const activePaperMeta = document.getElementById("activePaperMeta");


// --------------------------------------------------
// API
// --------------------------------------------------

const API_BASE_URL = "http://127.0.0.1:8000";

const ASK_URL = `${API_BASE_URL}/ask`;
const UPLOAD_URL = `${API_BASE_URL}/upload`;


// --------------------------------------------------
// Current selected file
// --------------------------------------------------

let selectedPdf = null;


// --------------------------------------------------
// Utility
// --------------------------------------------------

function formatFileSize(bytes) {

    if (bytes < 1024) {
        return `${bytes} B`;
    }

    if (bytes < 1024 * 1024) {
        return `${(bytes / 1024).toFixed(1)} KB`;
    }

    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}


function escapeHtml(value) {

    return value
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}


// --------------------------------------------------
// Error handling
// --------------------------------------------------

function showError(message) {

    errorMessage.textContent = message;

    error.classList.remove("hidden");

    result.classList.add("hidden");
}


function hideError() {

    error.classList.add("hidden");
}


// --------------------------------------------------
// Select PDF
// --------------------------------------------------

function handleFileSelection(file) {

    hideError();

    if (!file) {
        return;
    }


    if (
        file.type !== "application/pdf" &&
        !file.name.toLowerCase().endsWith(".pdf")
    ) {

        showError(
            "Please choose a PDF research paper."
        );

        return;
    }


    selectedPdf = file;

    fileName.textContent = file.name;

    fileSize.textContent =
        `${formatFileSize(file.size)} · Ready to upload`;


    selectedFile.classList.remove("hidden");

    uploadButton.classList.remove("hidden");

    dropZone.classList.add("hidden");
}


// --------------------------------------------------
// Choose file button
// --------------------------------------------------

chooseFileButton.addEventListener(
    "click",
    function (event) {

        event.stopPropagation();

        fileInput.click();
    }
);


// --------------------------------------------------
// Clicking drop zone
// --------------------------------------------------

dropZone.addEventListener(
    "click",
    function () {

        fileInput.click();
    }
);


// --------------------------------------------------
// File input
// --------------------------------------------------

fileInput.addEventListener(
    "change",
    function () {

        const file = fileInput.files[0];

        handleFileSelection(file);
    }
);


// --------------------------------------------------
// Drag and drop
// --------------------------------------------------

dropZone.addEventListener(
    "dragover",
    function (event) {

        event.preventDefault();

        dropZone.classList.add("dragover");
    }
);


dropZone.addEventListener(
    "dragleave",
    function () {

        dropZone.classList.remove("dragover");
    }
);


dropZone.addEventListener(
    "drop",
    function (event) {

        event.preventDefault();

        dropZone.classList.remove("dragover");

        const file = event.dataTransfer.files[0];

        handleFileSelection(file);
    }
);


// --------------------------------------------------
// Remove selected file
// --------------------------------------------------

removeFileButton.addEventListener(
    "click",
    function () {

        selectedPdf = null;

        fileInput.value = "";

        selectedFile.classList.add("hidden");

        uploadButton.classList.add("hidden");

        dropZone.classList.remove("hidden");
    }
);


// --------------------------------------------------
// Upload PDF
// --------------------------------------------------

async function uploadPdf() {

    if (!selectedPdf) {

        showError(
            "Please choose a PDF first."
        );

        return;
    }


    hideError();

    uploadButton.disabled = true;

    uploadButtonText.textContent =
        "Processing...";


    uploadStatus.classList.remove("hidden");

    uploadSpinner.classList.remove("hidden");

    uploadStatusTitle.textContent =
        "Processing paper";


    uploadStatusText.textContent =
        "Extracting text, creating embeddings, and building the research index...";


    try {

        const formData = new FormData();

        formData.append(
            "file",
            selectedPdf
        );


        const response = await fetch(
            UPLOAD_URL,
            {
                method: "POST",
                body: formData
            }
        );


        const data = await response.json();


        if (!response.ok) {

            throw new Error(
                data.detail ||
                `Upload failed with HTTP ${response.status}.`
            );
        }


        // ------------------------------------------
        // Upload succeeded
        // ------------------------------------------

        uploadSpinner.classList.add("hidden");

        uploadStatusTitle.textContent =
            "Paper processed successfully";


        uploadStatusText.textContent =
            `${data.pages} pages · ${data.chunks} chunks indexed`;


        activePaperName.textContent =
            data.filename;


        activePaperMeta.textContent =
            `${data.pages} pages · Ready`;


        activePaper.classList.remove("hidden");


        uploadButton.classList.add("hidden");


        selectedFile.classList.add("hidden");

    }

    catch (error) {

        console.error(error);

        uploadStatus.classList.add("hidden");

        showError(
            error.message ||
            "Could not upload the research paper."
        );

        uploadButton.disabled = false;

        uploadButtonText.textContent =
            "Upload & Process Paper";
    }
}


uploadButton.addEventListener(
    "click",
    uploadPdf
);


// --------------------------------------------------
// Loading state for questions
// --------------------------------------------------

function setLoading(isLoading) {

    if (isLoading) {

        loading.classList.remove("hidden");

        askButton.disabled = true;

        buttonText.textContent =
            "Researching...";

        result.classList.add("hidden");

        error.classList.add("hidden");

    } else {

        loading.classList.add("hidden");

        askButton.disabled = false;

        buttonText.textContent =
            "Ask ResearchRAG";
    }
}


// --------------------------------------------------
// Display answer
// --------------------------------------------------

function displayAnswer(data) {

    answerElement.textContent =
        data.answer ||
        "No answer was returned.";


    sourcesElement.innerHTML = "";


    if (
        !data.sources ||
        data.sources.length === 0
    ) {

        sourcesElement.innerHTML = `
            <div class="source-item">
                <div class="source-name">
                    No source information was returned.
                </div>
            </div>
        `;

    } else {

        data.sources.forEach(
            function (source) {

                const sourceElement =
                    document.createElement("div");

                sourceElement.className =
                    "source-item";


                sourceElement.innerHTML = `

                    <div class="source-top">

                        <span class="source-evidence">
                            ${escapeHtml(
                                source.evidence_id
                            )}
                        </span>

                        <span class="source-page">
                            Page ${escapeHtml(
                                String(source.page)
                            )}
                        </span>

                    </div>

                    <div class="source-name">
                        ${escapeHtml(
                            source.source
                        )}
                    </div>

                `;


                sourcesElement.appendChild(
                    sourceElement
                );
            }
        );
    }


    result.classList.remove("hidden");

    error.classList.add("hidden");


    result.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });
}


// --------------------------------------------------
// Ask ResearchRAG
// --------------------------------------------------

async function askResearchRAG() {

    const question =
        questionInput.value.trim();


    if (!question) {

        showError(
            "Please enter a question first."
        );

        questionInput.focus();

        return;
    }


    setLoading(true);


    try {

        const response = await fetch(
            ASK_URL,
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    question: question
                })
            }
        );


        if (!response.ok) {

            throw new Error(
                `ResearchRAG returned HTTP ${response.status}.`
            );
        }


        const data =
            await response.json();


        displayAnswer(data);

    }

    catch (error) {

        console.error(error);

        showError(
            "Could not connect to the ResearchRAG API. " +
            "Make sure the FastAPI server is running."
        );

    }

    finally {

        setLoading(false);
    }
}


// --------------------------------------------------
// Ask button
// --------------------------------------------------

askButton.addEventListener(
    "click",
    askResearchRAG
);


// --------------------------------------------------
// Ctrl + Enter / Cmd + Enter
// --------------------------------------------------

questionInput.addEventListener(
    "keydown",
    function (event) {

        if (
            event.key === "Enter" &&
            (event.ctrlKey || event.metaKey)
        ) {

            event.preventDefault();

            askResearchRAG();
        }
    }
);