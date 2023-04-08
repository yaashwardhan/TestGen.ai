document.addEventListener("DOMContentLoaded", function() {
    const slider = document.getElementById("durationOfTest");
    const sliderValue = document.getElementById("slider-value2");
    const textarea = document.getElementById("context");
  
    function updateSliderValue() {
      const value = Math.round(slider.value);
      sliderValue.innerHTML = value;
      document.getElementById("durationOfTest").innerHTML = value;
    }
  
    slider.addEventListener("input", updateSliderValue);
    textarea.addEventListener("input", function() {
      updateWordCount();
      updateSliderMax();
      updateSliderValue();
    });
  
    function updateWordCount() {
      const wordCount = textarea.value.trim().split(/\s+/).length;
      document.getElementById("word-count").innerHTML = wordCount;
    }
  
    updateWordCount();
    updateSliderMax();
    updateSliderValue();
  });
  