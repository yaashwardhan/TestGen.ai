// Examples:
// AJAX (Asynchronous JavaScript and XML) is a technique used in web development that allows web pages to update content dynamically without requiring a page reload. AJAX allows web applications to send and receive data asynchronously between the client and server without interfering with the display and behavior of the existing page.

$(document).ready(function() {
  $('form').submit(function(event) {
    event.preventDefault();
    var context = $('#context').val();
    var questionCount = $('#question-count').val();
    var incorrectOptionsCount = $('#incorrect-options-count').val();
    
    // Show loading screen
    $('#loading-screen').show();

    $.ajax({
      type: 'POST',
      url: '/generate-questions',
      contentType: 'application/json',
      data: JSON.stringify({
        'context': context,
        'questionCount': questionCount,
        'incorrectOptionsCount': incorrectOptionsCount
      }),
      success: function(response) {
        var questionsHtml = response['questions'].replace(/\\n/g, '<br>').replace(/"/g, '');
        $('#questions').html(questionsHtml);
        console.log(response);
        
        // Hide loading screen
        $('#loading-screen').hide();
      },
      error: function(xhr, status, error) {
        console.log(xhr.responseText);
        
        // Hide loading screen
        $('#loading-screen').hide();
      }
    });
  });
});
