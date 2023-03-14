function validateRange() {
    var rangeValue = document.getElementById("question-count").value;
    if (rangeValue == 0) {
      alert("Please choose a question count greater than 0.");
      return false; // Prevent form submission
    }
    return true; // Allow form submission
  }