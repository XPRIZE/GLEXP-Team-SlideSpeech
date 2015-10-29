/*
 *  Global variable naming convention: g_variableName
 */
var g_start = new Date();
var g_maxTime = 4000;
var g_timeoutVal = Math.floor(g_maxTime/100);
var g_timeoutVar;
var g_onGIF = "";
var g_referenceImage;
var g_refGreyCount = 0;
var g_threshold = 0;

// Moved to separate files
// var g_audioElement01 = document.createElement('audio');
// g_audioElement01.setAttribute('src', 'assets/audio/lower.mp3');

function updateProgress(percentage) {
  document.getElementById("pbar_innerdiv").style.width = percentage + "%";
}

function animateUpdate() {
  var now = new Date();
  var timeDiff = now.getTime() - g_start.getTime();
  var perc = 100 - Math.round((timeDiff/g_maxTime)*100);
  if (perc > 0) {
    updateProgress(perc);
    g_timeoutVar = setTimeout(animateUpdate, g_timeoutVal);
  } else {
    updateProgress(0);
    loop(); 
    if (g_onGIF.length == 0) {
      $.get("http://localhost:5000/wiped/no_gif")
    } else {
      $.get("http://localhost:5000/wiped/"+g_onGIF)
    }
  }
}

// Variables for referencing the canvas and 2dcanvas context
var g_canvas, g_ctx;

// Variables to keep track of the mouse position and left-button status 
var g_mouseX, g_mouseY, g_mouseDown=0;

// Variables to keep track of the touch position
var g_touchX, g_touchY;

// Draws a dot at a specific position on the supplied canvas name
// Parameters are: A canvas context, the x position, the y position, the size of the dot
function drawDot(ctx, x, y, size) {

  // Let's use black by setting RGB values to 0, and 255 alpha (completely opaque)
  var r = 255; 
  var g = 0; 
  var b = 0; 
  var a = 255;

  // Select a fill style
  ctx.fillStyle = "rgba("+r+","+g+","+b+","+(a/255)+")";

  // Draw a filled circle
  ctx.beginPath();
  ctx.arc(x, y, size, 0, Math.PI*2, true); 
  ctx.closePath();
  ctx.fill();
} 

// Keep track of the mouse button being pressed and draw a dot at current location
function sketchpad_mouseDown() {
  g_mouseDown = 1;
  drawDot(g_ctx, g_mouseX, g_mouseY, 12);
}

// Keep track of the mouse button being released
function sketchpad_mouseUp(e) {
  // Update the mouse co-ordinates when released
  getMousePos(e);
  console.log("Up at", g_mouseX, g_mouseY);
  if (g_mouseX > g_canvas.width || g_mouseY > g_canvas.height || (g_mouseX < 65 && g_mouseY < 45)) {
    return;
  }
  if (g_mouseX > 386 && g_mouseY < 20) {

    if (g_onGIF.length == 0) {
      $.get("http://localhost:5000/clearCanvas/no_gif")
    } else {
      $.get("http://localhost:5000/clearCanvas/" + g_onGIF)
    }
    g_mouseDown = 0;
    g_ctx.clearRect(0, 0, sketchpad.width, sketchpad.height);
    if (g_onGIF === "") {
      draw_background();
    } else {
      draw_example(g_onGIF);
    }
    updateProgress(0);
    clearTimeout(g_timeoutVar);
    return;
  }
  if (g_onGIF.length == 0) {
    $.get("http://localhost:5000/sketchpad_mouseUp/no_gif")
  } else {
    $.get("http://localhost:5000/sketchpad_mouseUp/"+g_onGIF)
  }

  // get current image
  var myImageData = g_ctx.getImageData(0, 0, g_canvas.width, g_canvas.height);
  var data = myImageData.data;
  if (g_onGIF !== "") {
    // Try to assess correctness
    var ref_data = g_referenceImage.data;

    // turn any "good" red to black
    for (var i = 0; i < data.length; i += 4) {
      if (data[i] != ref_data[i]) {
	data[i]     = 0; //255 - data[i];     // red
	data[i + 1] = 0; //255 - data[i + 1]; // green
	data[i + 2] = 0; //255 - data[i + 2]; // blue
      }
    }
    g_ctx.putImageData(myImageData, 0, 0);

    // count up red and grey
    var red_count = 0;
    var grey_count = 0;
    for (i = 0; i < data.length; i += 4) {
      if (data[i] === 255 && data[i + 1] === 0) { 
	red_count += 1;
      }
      if (data[i] !== 255 && data[i] !== 0) { 
	grey_count += 1;
      }
    }
    if (red_count < g_threshold && grey_count < g_threshold) {
      console.log("Red/Grey", red_count, grey_count, "PASS");
      document.getElementById('gif').style.display = 'none';
      g_onGIF = "";
    } else {
      console.log("Red/Grey", red_count, grey_count);
    }
  }

  g_mouseDown = 0;
  clearTimeout(g_timeoutVar);
  g_start = new Date();
  animateUpdate();
}

function loop() {
  /// clear canvas, set alpha and re-draw image
  g_ctx.clearRect(0, 0, sketchpad.width, sketchpad.height);
  if (g_onGIF === "") {
    draw_background();
  } else {
    draw_example(g_onGIF);
  }
}

function draw_background() {
  var img = new Image();
  img.setAttribute('crossOrigin', 'anonymous');
  img.onload = function(){
    g_ctx.drawImage(img,-3,0);
  };
  img.src = 'assets/images/background.png' + '?' + new Date().getTime();
}


// Keep track of the mouse position and draw a dot if mouse button is currently pressed
function sketchpad_mouseMove(e) { 

  // Update the mouse co-ordinates when moved
  getMousePos(e);

  // Draw a dot if the mouse button is currently being pressed
  if (g_mouseDown==1) {
    drawDot(g_ctx, g_mouseX, g_mouseY, 12);
  }
}

// Get the current mouse position relative to the top-left of the canvas
function getMousePos(e) {
  if (!e)
    e = event;

  if (e.offsetX) {
    g_mouseX = e.offsetX;
    g_mouseY = e.offsetY;
  }
  else if (e.layerX) {
    g_mouseX = e.layerX;
    g_mouseY = e.layerY;
  }
}

// Draw something when a touch start is detected
function sketchpad_touchStart() {

  // Update the touch co-ordinates
  getTouchPos();

  drawDot(g_ctx, g_touchX, g_touchY, 12);

  // Prevents an additional mousedown event being triggered
  event.preventDefault();
}

// Draw something and prevent the default scrolling when touch movement is detected
function sketchpad_touchMove(e) { 

  // Update the touch co-ordinates
  getTouchPos(e);

  // During a touchmove event, unlike a mousemove event, we don't need to check if the touch 
  // is engaged, since there will always be contact with the screen by definition.
  drawDot(g_ctx, g_touchX, g_touchY, 12); 

  // Prevent a scrolling action as a result of this touchmove triggering.
  event.preventDefault();
}

// Get the touch position relative to the top-left of the canvas
// When we get the raw values of pageX and pageY below, they take into account the scrolling on the page
// but not the position relative to our target div. We'll adjust them using "target.offsetLeft" and
// "target.offsetTop" to get the correct values in relation to the top left of the canvas.
function getTouchPos(e) {
  if (!e)
    e = event;

  if(e.touches) {
    if (e.touches.length == 1) { // Only deal with one finger
      var touch = e.touches[0]; // Get the information for finger #1
      g_touchX = touch.pageX - touch.target.offsetLeft;
      g_touchY = touch.pageY - touch.target.offsetTop;
    }
  }
}

// Set-up the canvas and add our event handlers after the page has loaded
function init() {

  // Get the specific canvas element from the HTML document
  g_canvas = document.getElementById('sketchpad');

  // If the browser supports the canvas tag, get the 2d drawing context for this canvas
  if (g_canvas.getContext)
    g_ctx = g_canvas.getContext('2d');

  // Check that we have a valid context to draw on/with before adding event handlers
  if (g_ctx) {

    // React to mouseup on the entire document
    window.addEventListener('mouseup', sketchpad_mouseUp, false);

    // React to mouse events on the canvas
    g_canvas.addEventListener('mousedown', sketchpad_mouseDown, false);
    g_canvas.addEventListener('mousemove', sketchpad_mouseMove, false);

    // React to touch events on the canvas
    g_canvas.addEventListener('touchstart', sketchpad_touchStart, false);
    g_canvas.addEventListener('touchmove', sketchpad_touchMove, false);
  }
}


function show_gif(gif) {
  $.get("http://localhost:5000/show_gif/"+gif)
  g_onGIF = gif;
  document.getElementById('gif').style.display = "block";
  document.getElementById('gif').innerHTML = 
    '<div style="float:left;"><img onclick="dismiss_demo();" src="assets/images/' + gif +
    '.gif"></div><div style="float:left; margin-left:10px"><H1><i onclick="play_audio(\'' + gif +
    '\');" class="fa fa-play"></i></H1></div></div><br style="clear:both">';
  draw_example(gif);
}

function dismiss_demo() {
  document.getElementById('gif').style.display = "none";
  g_onGIF = "";
}

function play_audio(mp3) {
  $.get("http://localhost:5000/play_audio/"+mp3)
  console.log(mp3);
  g_audioElement01.setAttribute('src', 'assets/audio/' + mp3 + '.mp3');
  g_audioElement01.play();
}

function draw_example(gif){
  document.getElementById('gif').style.display = "block";
  clearTimeout(g_timeoutVar);
  updateProgress(0);

  var img = new Image();
  img.setAttribute('crossOrigin', 'anonymous');
  img.onload = function(){
    g_ctx.drawImage(img,-3,-5);
    g_referenceImage = g_ctx.getImageData(0, 0, g_canvas.width, g_canvas.height);

    var ref_data = g_referenceImage.data;

    g_refGreyCount = 0;
    for (var i = 0; i < ref_data.length; i += 4) {
      if (ref_data[i] !== 255 && ref_data[i] !== 0 ) { 
        g_refGreyCount += 1;
      }
    }
    g_threshold = 0.1 * g_refGreyCount;
    console.log("Ref", g_refGreyCount, g_threshold);
  };
  img.src = 'assets/images/' + gif + '_.gif' + '?' + new Date().getTime();
  $('html, body').animate({ scrollTop: 0 }, 'fast');
}


