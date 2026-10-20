/**
 * Kisan Drishti - Main Application Controller
 * Handles Camera, File Uploads, Bilingual UI Switcher & Diagnosis Render
 * Author: Khushi Yadav
 */

document.addEventListener("DOMContentLoaded", () => {
  let currentLang = "hi"; // Default to Hindi for Indian farmers
  let currentStream = null;
  let activeSampleTag = null;
  let lastPrediction = null;

  // DOM Elements
  const btnLangHi = document.getElementById("btnLangHi");
  const btnLangEn = document.getElementById("btnLangEn");
  const tabCamera = document.getElementById("tabCamera");
  const tabUpload = document.getElementById("tabUpload");
  const tabSamples = document.getElementById("tabSamples");
  const paneCamera = document.getElementById("paneCamera");
  const paneUpload = document.getElementById("paneUpload");
  const paneSamples = document.getElementById("paneSamples");
  const cameraStream = document.getElementById("cameraStream");
  const laserScanner = document.getElementById("laserScanner");
  const btnCapture = document.getElementById("btnCapture");
  const btnStartCam = document.getElementById("btnStartCam");
  const fileInput = document.getElementById("fileInput");
  const dropzone = document.getElementById("dropzone");
  const previewBox = document.getElementById("previewBox");
  const previewImg = document.getElementById("previewImg");
  const resultCard = document.getElementById("resultCard");
  const resultPlaceholder = document.getElementById("resultPlaceholder");
  const resultContainer = document.getElementById("resultContainer");
  const ttsButton = document.getElementById("ttsButton");

  // Sample Leaf buttons
  const sampleCards = document.querySelectorAll(".sample-card");

  // ===================== Language Switching =====================
  const i18n = {
    hi: {
      appName: "किसान दृष्टि",
      appSubtitle: "फसल रोग पहचान एवं कृषि सलाह प्रणाली",
      helpline: "किसान हेल्पलाइन: 1800-180-1551 (टोल-फ्री)",
      heroTitle: "अपनी फसल का पत्ता स्कैन करें और तुरंत सही इलाज पाएं",
      heroSubtitle: "मोबाइल कैमरा या फोटो से पत्ती के रोग की पहचान करें (38 फसल रोग समर्थित)",
      tempLabel: "तापमान",
      humidityLabel: "आर्द्रता (उमस)",
      sprayLabel: "छिड़काव सलाह",
      spraySafe: "✅ आज छिड़काव के लिए अनुकूल",
      tabCamTitle: "📸 कैमरा स्कैनर",
      tabUpTitle: "📁 फोटो अपलोड करें",
      tabSamTitle: "🧪 टेस्ट पत्तियां",
      camGuide: "पत्ते को फ्रेम के बीच में रखें और साफ फोटो लें",
      btnStartCamera: "कैमरा शुरू करें",
      btnCapture: "पत्ती स्कैन करें",
      dragDropText: "पत्ती की फोटो यहां खींचें या क्लिक करके चुनें",
      readyToAnalyze: "तैयार: पत्ती का विश्लेषण किया जा रहा है...",
      analyzingWait: "कृत्रिम बुद्धिमत्ता (AI) द्वारा पत्ती की जांच हो रही है...",
      listenVoice: "🔊 आवाज में सुनें",
      stopVoice: "⏹️ आवाज रोकें",
      confidenceText: "पहचान सटीकता:",
      secSymptoms: "🔍 रोग के मुख्य लक्षण",
      secOrganic: "🌿 जैविक / प्राकृतिक उपचार (सस्ता व सुरक्षित)",
      secChemical: "🧪 रासायनिक कीटनाशक / कवकनाशी (सटीक मात्रा)",
      secPrevention: "🛡️ रोकथाम और सावधानियां (फसल चक्र व जल प्रबंधन)",
      secFertilizer: "🌾 पोषण एवं उर्वरक सलाह",
      topDifferential: "अन्य संभावित रोग (Top-3 तुलना):",
      placeholderText: "रोग निदान एवं उपचार सलाह देखने के लिए बाएं पैनल से पत्ती स्कैन करें या टेस्ट पत्ती चुनें।",
      footerProject: "बी.टेक फाइनल ईयर प्रोजेक्ट • प्लांटविलेज डेटासेट पर प्रशिक्षित MobileNetV2 मॉडल",
    },
    en: {
      appName: "Kisan Drishti",
      appSubtitle: "AI Crop Disease Detection & Agronomy Advisory",
      helpline: "Kisan Helpline: 1800-180-1551 (Toll-Free)",
      heroTitle: "Scan crop leaf to detect diseases & get instant cures",
      heroSubtitle: "Identify plant diseases early via camera or photo (38 classes supported)",
      tempLabel: "Temperature",
      humidityLabel: "Humidity",
      sprayLabel: "Spray Advisory",
      spraySafe: "✅ Safe to Spray Today",
      tabCamTitle: "📸 Live Camera",
      tabUpTitle: "📁 Upload Photo",
      tabSamTitle: "🧪 Test Samples",
      camGuide: "Center the leaf inside the frame and take a clear photo",
      btnStartCamera: "Start Camera",
      btnCapture: "Scan Leaf",
      dragDropText: "Drop leaf image here or click to browse",
      readyToAnalyze: "Ready: Leaf image loaded for analysis",
      analyzingWait: "AI Neural Network analyzing leaf foliage...",
      listenVoice: "🔊 Listen Advice",
      stopVoice: "⏹️ Stop Audio",
      confidenceText: "Model Confidence:",
      secSymptoms: "🔍 Characteristic Symptoms",
      secOrganic: "🌿 Organic & Bio-Control Remedies",
      secChemical: "🧪 Chemical Treatments & Precise Dosage",
      secPrevention: "🛡️ Cultural Practices & Preventative Measures",
      secFertilizer: "🌾 Fertilizer & Micronutrient Strategy",
      topDifferential: "Differential Diagnosis (Top-3 Predictions):",
      placeholderText: "Scan a leaf using your camera, upload an image, or click a test sample on the left to view instant diagnosis & treatment advice.",
      footerProject: "B.Tech Final Year Capstone Project • Fine-Tuned MobileNetV2 on PlantVillage Benchmark",
    }
  };

  function updateLanguage(lang) {
    currentLang = lang;
    btnLangHi.classList.toggle("active", lang === "hi");
    btnLangEn.classList.toggle("active", lang === "en");

    const t = i18n[lang];
    document.querySelectorAll("[data-i18n]").forEach(el => {
      const key = el.getAttribute("data-i18n");
      if (t[key]) el.textContent = t[key];
    });

    if (lastPrediction) {
      renderDiagnosis(lastPrediction);
    }
  }

  btnLangHi.addEventListener("click", () => updateLanguage("hi"));
  btnLangEn.addEventListener("click", () => updateLanguage("en"));

  // ===================== Tab Switching =====================
  function switchTab(tab) {
    [tabCamera, tabUpload, tabSamples].forEach(t => t.classList.remove("active"));
    [paneCamera, paneUpload, paneSamples].forEach(p => p.classList.remove("active"));

    if (tab === "camera") {
      tabCamera.classList.add("active");
      paneCamera.classList.add("active");
      startCamera();
    } else if (tab === "upload") {
      tabUpload.classList.add("active");
      paneUpload.classList.add("active");
      stopCamera();
    } else if (tab === "samples") {
      tabSamples.classList.add("active");
      paneSamples.classList.add("active");
      stopCamera();
    }
  }

  tabCamera.addEventListener("click", () => switchTab("camera"));
  tabUpload.addEventListener("click", () => switchTab("upload"));
  tabSamples.addEventListener("click", () => switchTab("samples"));

  // ===================== Camera Handling =====================
  async function startCamera() {
    try {
      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        currentStream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: { ideal: "environment" }, width: { ideal: 640 }, height: { ideal: 480 } }
        });
        cameraStream.srcObject = currentStream;
        btnStartCam.style.display = "none";
        btnCapture.disabled = false;
      }
    } catch (err) {
      console.warn("Camera access unavailable, defaulting to upload/samples:", err);
      btnStartCam.style.display = "inline-flex";
      btnStartCam.textContent = currentLang === "hi" ? "कैमरा अनुमति दें" : "Grant Camera Access";
    }
  }

  function stopCamera() {
    if (currentStream) {
      currentStream.getTracks().forEach(track => track.stop());
      currentStream = null;
    }
  }

  btnStartCam.addEventListener("click", startCamera);

  btnCapture.addEventListener("click", () => {
    if (!cameraStream.videoWidth) return;
    const canvas = document.createElement("canvas");
    canvas.width = cameraStream.videoWidth;
    canvas.height = cameraStream.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(cameraStream, 0, 0, canvas.width, canvas.height);

    previewImg.src = canvas.toDataURL("image/jpeg");
    previewBox.style.display = "flex";
    activeSampleTag = null;
    triggerInference(previewImg);
  });

  // ===================== File Upload Handling =====================
  dropzone.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.style.borderColor = "var(--primary)";
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.style.borderColor = "#86efac";
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.style.borderColor = "#86efac";
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  });

  function handleFile(file) {
    if (!file.type.startsWith("image/")) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      previewBox.style.display = "flex";
      activeSampleTag = null;
      triggerInference(previewImg);
    };
    reader.readAsDataURL(file);
  }

  // ===================== Sample Cards =====================
  sampleCards.forEach(card => {
    card.addEventListener("click", () => {
      const tag = card.getAttribute("data-sample");
      const img = card.querySelector("img");
      previewImg.src = img.src;
      previewBox.style.display = "flex";
      activeSampleTag = tag;
      triggerInference(previewImg, tag);
    });
  });

  // ===================== Inference Trigger =====================
  async function triggerInference(imgElement, sampleTag = null) {
    laserScanner.style.display = "block";
    resultPlaceholder.style.display = "block";
    resultPlaceholder.querySelector("p").textContent = i18n[currentLang].analyzingWait;
    resultContainer.style.display = "none";

    // Wait until image is decoded
    if (!imgElement.complete) {
      await new Promise(r => imgElement.onload = r);
    }

    const prediction = await window.cropClassifier.predict(imgElement, sampleTag || activeSampleTag);
    lastPrediction = prediction;
    laserScanner.style.display = "none";
    renderDiagnosis(prediction);
  }

  // ===================== Render Diagnosis Card =====================
  function renderDiagnosis(pred) {
    const agronomy = AGRONOMY_DB[pred.topClass] || AGRONOMY_DB["Tomato___Late_blight"];
    resultPlaceholder.style.display = "none";
    resultContainer.style.display = "block";

    const isHi = currentLang === "hi";

    // Title & Badges
    const diseaseName = isHi ? agronomy.name_hi : agronomy.name_en;
    const cropName = isHi ? agronomy.crop_hi : agronomy.crop_en;
    const severityText = isHi ? agronomy.severity_hi : agronomy.severity;

    document.getElementById("diagDiseaseName").textContent = diseaseName;
    document.getElementById("diagPathogen").textContent = `${isHi ? 'रोगजनक:' : 'Pathogen:'} ${agronomy.scientific_name} • ${isHi ? agronomy.category_hi : agronomy.category}`;

    const badge = document.getElementById("diagSeverityBadge");
    badge.textContent = `${cropName} • ${severityText}`;
    badge.className = "status-badge " + (agronomy.severity.includes("Critical") || agronomy.severity.includes("High") ? "badge-critical" : (agronomy.severity.includes("Healthy") ? "badge-healthy" : "badge-warning"));

    // Confidence
    const confPct = (pred.confidence * 100).toFixed(1);
    document.getElementById("diagConfidenceVal").textContent = `${confPct}%`;
    document.getElementById("diagConfidenceFill").style.width = `${confPct}%`;

    // Advisory blocks
    document.getElementById("txtSymptoms").textContent = isHi ? agronomy.symptoms_hi : agronomy.symptoms_en;
    document.getElementById("txtOrganic").textContent = isHi ? agronomy.organic_treatment_hi : agronomy.organic_treatment_en;
    document.getElementById("txtChemical").textContent = isHi ? agronomy.chemical_treatment_hi : agronomy.chemical_treatment_en;
    document.getElementById("txtPrevention").textContent = isHi ? agronomy.prevention_hi : agronomy.prevention_en;
    document.getElementById("txtFertilizer").textContent = isHi ? agronomy.fertilizer_advice_hi : agronomy.fertilizer_advice_en;

    // Top-3 differential predictions
    const diffContainer = document.getElementById("differentialList");
    diffContainer.innerHTML = "";
    pred.top3.forEach(item => {
      const itemAgro = AGRONOMY_DB[item.className];
      const itemName = itemAgro ? (isHi ? itemAgro.name_hi : itemAgro.name_en) : item.className;
      const pct = (item.confidence * 100).toFixed(1);

      const row = document.createElement("div");
      row.className = "diff-row";
      row.innerHTML = `
        <span style="width: 45%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${itemName}</span>
        <div class="diff-bar"><div class="diff-fill" style="width: ${pct}%"></div></div>
        <span style="font-weight: 700; width: 45px; text-align: right;">${pct}%</span>
      `;
      diffContainer.appendChild(row);
    });

    // Voice Speech Assistant configuration
    const speechText = isHi
      ? `पहचाना गया रोग है: ${agronomy.name_hi}। गंभीरता: ${agronomy.severity_hi}। मुख्य लक्षण: ${agronomy.symptoms_hi}। जैविक उपचार: ${agronomy.organic_treatment_hi}। रासायनिक उपचार: ${agronomy.chemical_treatment_hi}।`
      : `Identified disease: ${agronomy.name_en}. Severity: ${agronomy.severity}. Symptoms: ${agronomy.symptoms_en}. Organic remedy: ${agronomy.organic_treatment_en}. Chemical cure: ${agronomy.chemical_treatment_en}.`;

    ttsButton.onclick = () => {
      window.voiceAssistant.toggle(speechText, currentLang, ttsButton);
    };

    window.voiceAssistant.onStateChange = (speaking) => {
      ttsButton.textContent = speaking ? (isHi ? "⏹️ आवाज रोकें" : "⏹️ Stop Audio") : (isHi ? "🔊 आवाज में सुनें" : "🔊 Listen Advice");
      ttsButton.classList.toggle("speaking", speaking);
    };
  }

  // Accordion Expand / Collapse
  document.querySelectorAll(".advise-block-header").forEach(header => {
    header.addEventListener("click", () => {
      const parent = header.parentElement;
      parent.classList.toggle("open");
    });
  });

  // Open first two accordions by default
  document.querySelectorAll(".advise-block")[0]?.classList.add("open");
  document.querySelectorAll(".advise-block")[1]?.classList.add("open");

  // Initialize with sample 1 by default for instant delight
  sampleCards[0].click();
});
