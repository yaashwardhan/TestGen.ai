document.addEventListener("DOMContentLoaded", function() {
    const slider = document.getElementById("question-count");
    const sliderValue = document.getElementById("slider-value");
    const textarea = document.getElementById("context");
  
    function updateSliderMax() {
      const wordCount = textarea.value.trim().split(/\s+/).length;
      if (wordCount <= 10) {
        slider.max = 0;
      }
      else if (wordCount > 10 && wordCount <= 20) {
        slider.max = 1;
      } 
      else if (wordCount > 20 && wordCount <= 40) {
        slider.max = 2;
      }
      else if (wordCount > 40 && wordCount <= 200) {
        slider.max = 3;
      } else {
        slider.max = 5;
      }
    }
  
    function updateSliderValue() {
      const value = Math.round(slider.value);
      sliderValue.innerHTML = value;
      document.getElementById("question-count").innerHTML = value;
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
  