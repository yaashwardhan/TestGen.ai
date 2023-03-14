document.addEventListener("DOMContentLoaded", function() {
    var textarea = document.getElementById("context");
    var wordCountDisplay = document.getElementById("word-count");
    function updateWordCount() {
      var wordCount = textarea.value.trim().split(/\s+/).length;
      wordCountDisplay.innerHTML = wordCount;
    }
    textarea.addEventListener("input", updateWordCount);
    updateWordCount();
});