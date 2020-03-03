$("#image-selector").change(function() { // get the image from image-selector element
    let reader = new FileReader();
    console.log(reader);
    reader.onload = function() {
        let dataURL = reader.result; //contain the image data as an URL for encoded the string 
        console.log(reader.result);
        console.log(dataURL);
        
        // source attribute of the selected image to the value of data URL
        $('#selected-image').attr("src", dataURL);
        $("#prediction-list").empty();
    }
    let file = $("#image-selector").prop('files')[0];
    console.log($("#image-selector").prop('files')[0]);
    reader.readAsDataURL(file);
    
});

let model; //create the model variable
//IFE function is the function that run as soon as it's define
(async function() {
    // model = await tf.loadLayersModel('http://192.168.16.78:81/tfjs-models/mobileNet/model.json');
    model = await tf.loadLayersModel('http://localhost:81/tfjs-models/VGG16/model.json');
    $('.progress-bar').hide();
    console.log(model); 
})();

// when the predict botton is clicked
$("#predict-button").click(async function(){
    let image =  $('#selected-image').get(0);
    console.log(image);
    // debugger
    let tensor = tf.browser.fromPixels(image)
        .resizeBilinear([224,224]) //resize imgae to (h,w) = 224x224
        .toFloat() // chage input image to float32
        .expandDims(); //expand the dimensions to be of ranf four
        console.log(tensor); 
        // debugger 
    // more pre-processing to be added here later

    let predictions = await model.predict(tensor).data(); //give the probability for an individual imagenet class
    //is made up of 1000 elements corresponding to the prediction prob for an individual imageNet classses
    let top5 = Array.from(predictions)
        .map(function (p,i){
            return {
                probability : p,
                className: IMAGENET_CLASSES[i]
            };
        }).sort(function(a,b){
            return b.probability - a.probability;
        }).slice(0,5);

    $("#prediction-list").empty();
    top5.forEach(function (p) { 
        $('#prediction-list').append(`<li>${p.className} : ${p.probability.toFixed(6)}</li>`);
    });
});

