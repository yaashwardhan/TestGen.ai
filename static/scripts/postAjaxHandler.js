// Examples:
// AJAX (Asynchronous JavaScript and XML) is a technique used in web development that allows web pages to update content dynamically without requiring a page reload. AJAX allows web applications to send and receive data asynchronously between the client and server without interfering with the display and behavior of the existing page.

$(document).ready(function() {
  $('form').submit(function(event) {
    event.preventDefault();
    var context = $('#context').val();
    var questionCount = $('#question-count').val();
    console.log(questionCount);

    // Show loading screen
    $('#loading-screen').show();
    $('#questions').html('');
    $.ajax({
      type: 'POST',
      url: '/generate-questions',
      contentType: 'application/json',
      data: JSON.stringify({
        'context': context,
        'questionCount': questionCount,
      }),
      success: function(response) {
        var questions = response['questions'];
        var correctAnswers = response['correctAnswers'];
        var distractor1 = response['distractor1'];
        var distractor2 = response['distractor2'];
        var distractor3 = response['distractor3'];
        var html = '';
        html += '<div class="top-container">';
        html += '<p class="smallheadings">Generated MCQ</p>'; 
        html += '</div>';
        html += '<div class="question-container-outer-parent">';
        html += '<div class="question-container-outer" style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;">';
        
        for (var i = 0; i < questions.length; i++) {
          var options = ['Ans: '+correctAnswers[i]].concat(distractor1[i]).concat(distractor2[i]).concat(distractor3[i]);
          html += '<div class="question-container">';
          html += '<div class="question">';
          html += '<div class="question-sidebyside" style="display: flex;align-items: center;"><span class="edit-icon" style="flex: 1;display: flex; justify-content: center; align-items: center;">&#9998;</span><p class="edit-question" style="flex: 7; " contenteditable>' + questions[i] + '</p><span class="delete-icon" style="flex: 1;display: flex; justify-content: center; align-items: center;">&#128465;</span></div>';
          html += '<ul class="options" data-question="' + i + '" data-group="' + i + '">';
          html += '<li class="option" style="color: greenyellow;" data-value="' + options[0] + '">' + "<span style='color:dimgrey;'>&#9776;</span>" +options[0]  + '<span class="remove-option" style="color:#CC5500;">&#8998;</span></li>';
          html += '<li class="option" data-value="' + options[1] + '">' + "<span style='color:dimgrey;'>&#9776;</span>" + options[1] +'<span class="remove-option" style="color:#CC5500;">&#8998;</span></li>';
          html += '<li class="option" data-value="' + options[2] + '">' + "<span style='color:dimgrey;'>&#9776;</span>" + options[2] +'<span class="remove-option" style="color:#CC5500;">&#8998;</span></li>';
          html += '<li class="option" data-value="' + options[3] + '">' + "<span style='color:dimgrey;'>&#9776;</span>" + options[3] +'<span class="remove-option" style="color:#CC5500;">&#8998;</span></li>';
          html += '<div class="custom-option"><input type="text" class="my-input" placeholder="Extra Choice"><span class="add-option">ADD</span></div>';
          html += '</ul>';
          html += '</div>';
          html += '</div>';
      }
      html += '<span class="add-question">ADD QUESTION</span>';
      html += '</div>';
      html += '</div>';
      
      $('#questions').html(html);
      // Add click event listener to plus button
      $(document).on('click', '.add-question', function() {
        var $container = $(this);
        var $questionContainer = $('<div>', {'class': 'question question-container'});
        var $question = $('<div>', {'class': 'question-sidebyside', 'style': 'display: flex;align-items: center;'});
        var $editIcon = $('<span>', {'class': 'edit-icon', 'style': 'flex: 1;display: flex; justify-content: center; align-items: center;'}).html('&#9998;');
        var $editQuestion = $('<p>', {'class': 'edit-question', 'style': 'flex: 7;', 'contenteditable': true}).text('Enter your question here');
        var $deleteIcon = $('<span>',{'class':"delete-icon" , 'style':"flex: 1;display: flex; justify-content: center; align-items: center;"}).html('&#128465;');
        var $options = $('<ul>', {'class': 'options', 'data-question': questions.length, 'data-group': questions.length});
        var $customOption = $('<div>', {'class': 'custom-option'});
        var $input = $('<input>', {'type': 'text', 'class': 'my-input', 'placeholder': 'Extra Choice'});
        var $addOption = $('<span>', {'class': 'add-option'}).text('ADD');

        $question.append($editIcon);
        $question.append($editQuestion);
        $question.append($deleteIcon);
        $questionContainer.append($question);

        $customOption.append($input);
        $customOption.append($addOption);
        $options.append($customOption);
        $questionContainer.append($options);
        

        $container.after($questionContainer);
        $('.question-container-outer').append($container);

        new Sortable($options.get(0), {
          group: questions.length,
          animation: 150,
          filter: '.add-option'
        });
      
        // Add click event listener to edit button/icon
        $(document).on('click', '.edit-button', function() {
          var $questionText = $(this).parent();
          $questionText.toggleClass('editing');
          $questionText.find('.edit-question').focus();
        });
      });


      console.log(questions);
      console.log(correctAnswers);
      console.log(distractor1);
      console.log(distractor2);
      console.log(distractor3);
      $('.options').each(function() {
          new Sortable(this, {
              group: $(this).data('group'),
              animation: 150,
              filter: '.add-option'
          });
      });

      
      // Remove option
      $(document).on('click', '.remove-option', function() {
        $(this).parent().remove();
      });

      // Add click event listener to delete button/icon
      $(document).on('click', '.delete-icon', function() {
        $(this).closest('.question-container').remove();
      });
  
      // Add custom option
      $(document).on('click', '.add-option', function() {
        var $input = $(this).prev('input');
        var value = $input.val();
        if (value.trim() !== '') {
          var $ul = $(this).parent().parent();
          var questionIndex = $ul.attr('data-question');
          var $li = $('<li>', {'class': 'option', 'data-value': value}).text(value);
          $li.prepend("<span style='color:dimgrey;'>&#9776;</span>");
          $li.append('<span class="remove-option" style="color:#CC5500;">&#8998;</span>');
          $ul.append($li);
          $input.val('');
        }
      });
      // Add click event listener to edit button/icon
      $(document).on('click', '.edit-button', function() {
        var $questionText = $(this).parent();
        $questionText.toggleClass('editing');
        $questionText.find('.edit-question').focus();
      });
      
        // Hide loading screen
        $('#loading-screen').hide();
      },
      error: function(xhr, status, error) {
        console.log(xhr.responseText);
        $('#loading-screen').hide();
      }
    });
  });
});
