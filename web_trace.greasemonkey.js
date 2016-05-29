// ==UserScript==
// @name        microcorruption Trace
// @namespace   zaddach.org
// @include     https://microcorruption.com/cpu/debugger
// @version     1
// @grant       none
// @require     https://raw.githubusercontent.com/eligrey/FileSaver.js/master/FileSaver.js
// ==/UserScript==

var TRACE_TIMEOUT = 10;

var do_trace_execution = false;

var trace_button = $("<a id='trace' class='button orange'>Trace execution</a>");
trace_button.appendTo($("#hideheaders").parent());

var trace_iframe = $("<iframe id='trace_iframe' />");
$('body').append(trace_iframe);
trace_iframe.hide();


function trace_step() {
  cpu.send("/cpu/step", {}, function(e) {
      console.log(JSON.stringify(e));
      trace_iframe.append(JSON.stringify(e) + "\n");
      //var doc = document.getElementById('trace_iframe').contentWindow.document;
      //doc.open();
      //doc.write(JSON.stringify(e) + "\n");
      //doc.close();
    
      cpu.do_update(e);
      if (do_trace_execution) {
        setTimeout(trace_step, TRACE_TIMEOUT); 
      }
    });
}

function toggle_trace_execution() {
  do_trace_execution = !do_trace_execution;
  if (do_trace_execution) {
    trace_button.text("Stop trace");
    setTimeout(trace_step, TRACE_TIMEOUT);
    setInterval(cpu.reset_expirey, 5 * 60 * 1000);
  }
  else {
    trace_button.text("Trace execution");
    var blob = new Blob([trace_iframe.html()], {type: "text/plain;charset=utf-8"});
    saveAs(blob, "trace.txt");
  }
}

trace_button.click(toggle_trace_execution);

