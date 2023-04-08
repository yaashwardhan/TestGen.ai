// $(document).ready(function() {
//     $('form').submit(function(event) {
//       event.preventDefault();
//       var context = $('#context').val();
//       var questionCount = $('#question-count').val();
//       var questionType = $('#question-count').val();
//       // Show loading screen
//       $('#loading-screen').show();
//       $('#questions').html('');
//       $.ajax({
//         type: 'POST',
//         url: '/generate-FIB',
//         contentType: 'application/json',
//         data: JSON.stringify({
//           'context': context,
//           'questionCount':questionCount,
//         }),
//         success: function(response) {
//           var html = '';
//           html += '<div class="top-container">';
//           html += '<p class="smallheadings">Generated Fill In The Blanks</p>'; 
//           html += '</div>';
//           html += '<div class="fib-question-container">';
      
//           for (var i = 0; i < response.length; i++) {
//             html += '<div class="fib-question">';
//             html += '<p>' + response[i]['question'] + '</p>';
//             html += '<p><strong>Answer:</strong> ' + response[i]['answer'] + '</p>';
//             html += '</div>';
//           }
//           html += '</div>';
        
//           $('#questions').html(html);
        
//           // Hide loading screen
//           $('#loading-screen').hide();
//         },
//         error: function(xhr, status, error) {
//           console.log(xhr.responseText);
//           $('#loading-screen').hide();
//         }
//       });
//     });
//   });
  