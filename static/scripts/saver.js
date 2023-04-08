    // Save questions to JSON file
    $('#save-button').click(function() {
        var questionsData = [];
        $('.question').each(function() {
          var question = $(this).find('p').text();
          var options = [];
          console.log('yes2');
          $(this).find('.option').each(function() {
            var option = $(this).text();
            options.push(option);
          });
  
          var questionData = {
            'question': question,
            'options': options
          };
  
          questionsData.push(questionData);
        });
        var extraData = {
            'title': $('#titleOfTest').val(),
            'correct': $('#slider-value').val(),
            'duration':$('#slider-value2').val(),
            // 'intro': $(this).find('h4').text(),
          };
        questionData.push(extraData);
        if (questionsData.length === 0) {
        alert("Please Generate Questions First!");
      } else {
        var json = JSON.stringify(questionsData, null, 2).replace(/\u2326/g, '').replace(/\u2630/g, '').replace(/\u270E/g,'');
        console.log(json);
        var blob = new Blob([json], {type: "application/json"});
        var url  = URL.createObjectURL(blob);
  
        var a = document.createElement('a');
        a.download = 'questions.json';
        a.href = url;
        a.textContent = 'Download questions';
        document.body.appendChild(a);
  
        // a.click();
  
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      }
    });