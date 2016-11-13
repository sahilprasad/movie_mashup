var cname ='scriptlysbion'
window.addEventListener('load', function(){
	function getCookie(cnme) {
    	var name = cnme + "=";
    	var ca = document.cookie.split(';');
    	for(var i = 0; i <ca.length; i++) {
        	var c = ca[i];
        	while (c.charAt(0)==' ') {
            	c = c.substring(1);
        	}
        	if (c.indexOf(name) == 0) {
				var el = document.getElementById("form-div")
				el.style.display = "block";
				el.style.textAlign = "center";
				el.style.left = 0;
				el.style.top = "210%";
				el.style.marginBottom = "15%"
				console.log("Got cookie")            	
            }
    	}
   	}

   	getCookie(cname)

	var num_images = Math.min((screen.width-100+25)/(175+25), 6)
	console.log(num_images)
	var margin_left = ((screen.width +25) -200 * (num_images + 1))/2
	var html = '<img src="showpic_1.jpg" height="250px" '
		+'width="175px" alt="movie poster 1" style="margin-top:20%; padding:25px; margin-left:' + margin_left + 'px;">'
	for(i = 1; i < num_images; i++) {
		var j = i + 1
		html += '<img src="showpic_' + j + '.jpg" height="250px" '
		+'width="175px" alt="movie poster '+ j + '" style="margin-top:20%; padding:25px">'
	}
	console.log(html)

	document.getElementById("show_pictures").innerHTML = html
	document.getElementById("show_pictures").margin = "auto"
	document.getElementById("show_pictures").display = "block"
	console.log(document.getElementById("show_pictures"))
	document.getElementById("generate").addEventListener("click", function(event) {
		var el = document.getElementById("form-div")
		el.style.display = "block";
		el.style.textAlign = "center";
		el.style.left = 0;
		el.style.top = "210%";
		el.style.marginBottom = "15%"
		// make a cookie
    	var d = new Date();
    	d.setTime(d.getTime() + (10000*24*60*60*1000));
    	var expires = "expires="+ d.toUTCString();
    	document.cookie = cname + "=doesnotmatter;" + expires + ";path=/";
	});
});