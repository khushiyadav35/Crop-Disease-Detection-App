/**
 * Kisan Drishti - Voice Speech Synthesis Assistant
 * Speaks diagnosis and remedies in Hindi (हिंदी) or English
 * Author: Khushi Yadav
 */

class VoiceAssistant {
  constructor() {
    this.synth = window.speechSynthesis || null;
    this.isSpeaking = false;
    this.voices = [];
    this.onStateChange = null;

    if (this.synth) {
      this.loadVoices();
      if (speechSynthesis.onvoiceschanged !== undefined) {
        speechSynthesis.onvoiceschanged = () => this.loadVoices();
      }
    }
  }

  loadVoices() {
    if (!this.synth) return;
    this.voices = this.synth.getVoices();
  }

  getBestVoice(lang = "hi") {
    if (!this.voices.length) this.loadVoices();
    if (lang === "hi") {
      const hiVoice = this.voices.find(v => v.lang.startsWith("hi") || v.lang.includes("Hindi"));
      if (hiVoice) return hiVoice;
    }
    // Fallback Indian English or English
    const enInVoice = this.voices.find(v => v.lang === "en-IN" || v.lang.includes("India"));
    if (enInVoice) return enInVoice;
    return this.voices.find(v => v.lang.startsWith("en")) || null;
  }

  speak(text, lang = "hi", callback) {
    if (!this.synth) {
      console.warn("Speech synthesis is not supported on this browser.");
      return;
    }

    this.stop();

    if (!text || text.trim() === "") return;

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = lang === "hi" ? "hi-IN" : "en-IN";
    utterance.rate = lang === "hi" ? 0.88 : 0.95; // Slightly slower pacing for elderly farmers
    utterance.pitch = 1.0;

    const voice = this.getBestVoice(lang);
    if (voice) utterance.voice = voice;

    utterance.onstart = () => {
      this.isSpeaking = true;
      if (this.onStateChange) this.onStateChange(true);
    };

    utterance.onend = () => {
      this.isSpeaking = false;
      if (this.onStateChange) this.onStateChange(false);
      if (callback) callback();
    };

    utterance.onerror = (e) => {
      console.warn("TTS error:", e);
      this.isSpeaking = false;
      if (this.onStateChange) this.onStateChange(false);
    };

    this.synth.speak(utterance);
  }

  stop() {
    if (this.synth && (this.synth.speaking || this.synth.pending)) {
      this.synth.cancel();
      this.isSpeaking = false;
      if (this.onStateChange) this.onStateChange(false);
    }
  }

  toggle(text, lang, btnElement) {
    if (this.isSpeaking) {
      this.stop();
      if (btnElement) btnElement.classList.remove("speaking");
    } else {
      if (btnElement) btnElement.classList.add("speaking");
      this.speak(text, lang, () => {
        if (btnElement) btnElement.classList.remove("speaking");
      });
    }
  }
}

window.voiceAssistant = new VoiceAssistant();
