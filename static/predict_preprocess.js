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

    // added pre-processing in image
    let meanImageNetRGB = { //mean of red green and blue channel in imageNet dataset refered to trainning mathod
        red: 123.68,
        green: 116.779,
        blue: 103.939
    };
    
    let indices = [ // create the list of 1D tensor of integers
        tf.tensor1d([0], "int32"), //contain the single value 0
        tf.tensor1d([1], "int32"), //contain the single value 1
        tf.tensor1d([2], "int32")
    ];

    // class object in Java
    let centeredRGB = {
        // red object that contain 1D tensor of center red values
        red: tf.gather(tensor, indices[0], 3) //gather red values from the tensor(at index 0) along the second axis (1,224,224,1)
        //of our 224x224x3 tensor
        .sub(tf.scalar(meanImageNetRGB.red)) // subtracting the mean imageNet red value from each red value in our tensor
        .reshape([50176]), //flattened image array into 1D array
        green: tf.gather(tensor, indices[1], 3)
        .sub(tf.scalar(meanImageNetRGB.green))
        .reshape([50176]),
        blue: tf.gather(tensor, indices[2], 3)
        .sub(tf.scalar(meanImageNetRGB.blue))
        .reshape([50176]),
    };

    let processedTensor = tf.stack([centeredRGB.red, centeredRGB.green, centeredRGB.blue], 1)
        .reshape([224,224,3])
        .reverse(2) // reverve the color from RGB to BGR
        .expandDims(); // expended dimension to rank 4

    // // easy way --> use broadcasting tensor
    // let image =  $('#selected-image').get(0);
    // let meanImageNetRGB = tf.tensor1d([123.68, 116.779, 103.939]);
    // let tensor = tf.fromPixels.browser(image)
    // .resizeBilinear([224,224])
    // .toFloat()
    // .sub(meanImageNetRGB)
    // .reverse(2)
    // .expandDims();
    
    
        // get prediction value
    let predictions = await model.predict(processedTensor).data(); //give the probability for an individual imagenet class 
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

