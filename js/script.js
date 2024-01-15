// variables
let mapInPanel;
let map;
let heatmap;
let clickedPolygon; // Variable to store the clicked polygon
let clickedPixelMarkers = []; // Array to store the clicked pixel markers
let allPolygons = []; // for hide/show
let segmentsButtonClicked = true;
let heatmapClicked = true;
let BinaryClicked = true

// Function to open the modal
function openModal(modalId) {
    // Input: modalId - ID of the modal to be opened
    // Output: None
    // Hide the other modal
    const otherModalId = modalId === 'infosModal' ? 'contactModal' : 'infosModal';
    document.getElementById(otherModalId).style.display = "none";

    // Show the selected modal
    document.getElementById(modalId).style.display = "block";
}

// Function to close the modal
function closeModal(modalId) {
    // Input: modalId - ID of the modal to be closed
    // Output: None
    document.getElementById(modalId).style.display = "none";
}

// Close the modal if the user clicks outside of it
window.onclick = function (event) {
    // Input: event - Click event
    // Output: None
    if (event.target.className === "modal") {
        event.target.style.display = "none";
    }
};

// Show only the Infos modal when the page is loaded
window.onload = function () {
    // Input: None
    // Output: None
    openModal('infosModal');
};

// Function to toggle the visibility of the segments
function toggleSegments() {
    // Input/Output: None

    // Toggle the variable indicating whether the segments button is clicked
    segmentsButtonClicked = !segmentsButtonClicked;

    // Get the segments button element
    const segmentsButton = document.getElementById('toggleSegmentsButton');

    // Update the font style based on whether the button is clicked
    if (segmentsButtonClicked) {
        segmentsButton.style.font = 'bold 16px Helvetica, sans-serif';
        segmentsButton.style.color = '#070707';
    } else {
        segmentsButton.style.font = ' 17px Helvetica, sans-serif';
        segmentsButton.style.color = '#7c7b7b';
    }
    allPolygons.forEach(polygon => {
        const currentVisibility = polygon.getMap() !== null;
        polygon.setMap(currentVisibility ? null : map);
    });
}

// Function to toggle the visibility of the heatmap
function toggleHeatmap() {
    // Input/Output: None

    // Toggle the variable indicating whether the segments button is clicked
    heatmapClicked = !heatmapClicked;

    // Get the segments button element
    const heatmap_button = document.getElementById('toggleHeatmapButton');

    // Update the font style based on whether the button is clicked
    if (heatmapClicked) {
        heatmap_button.style.font = 'bold 16px Helvetica, sans-serif';
        heatmap_button.style.color = '#070707';
    } else {
        heatmap_button.style.font = '17px Helvetica, sans-serif';
        heatmap_button.style.color = '#7c7b7b';
    }
    heatmap.setMap(heatmap.getMap() ? null : map);
}

// Function to toggle the visibility of the heatmap
function toggleBinary() {
    // Input/Output: None

    // Toggle the variable indicating whether the segments button is clicked
    BinaryClicked = !BinaryClicked;

    // Get the segments button element
    const binary_button = document.getElementById('toggleBinaryButton');

    // Update the font style based on whether the button is clicked
    if (BinaryClicked) {
        binary_button.style.font = 'bold 16px Helvetica, sans-serif';
        binary_button.style.color = '#070707';
    } else {
        binary_button.style.font = '17px Helvetica, sans-serif';
        binary_button.style.color = '#7c7b7b';
    }
}

// Function to dynamically convert alpha value to color
function getColorFromAlpha(alpha) {
    // Input: alpha - Alpha value
    // Output: Color value (string)

    if (BinaryClicked) {
        return alpha > 0.5 ? 'red' : 'blue';
    } else {
        // Convert alpha to a value between 0 and 255 for RGB
        let alphaScaled = Math.floor(alpha * 255);

        // Create RGB color
        let color = `rgb(${255 - alphaScaled}, 0, ${alphaScaled})`;

        return color;
    }
}

// Function to map class_id to a property (color or name)
function getClassProperty(classId, property) {
    // Input: classId - Class ID, property - 'color' or 'name'
    // Output: Color value or class name (string)

    const classProperties = {
        1: { color: "#f1c40f", name: "Cereals" },
        2: { color: "#a6acaf", name: "Cotton" },
        3: { color: "#2c3e50", name: "Oleag./Legum." },
        4: { color: "#5dade2", name: "Grassland" },
        5: { color: "#abebc6", name: "Shrubland" },
        6: { color: "#196f3d", name: "Forest" },
        7: { color: "#e74c3c", name: "Baresoil" },
        8: { color: "#162ef3", name: "Water" },
    };

    const defaultProperties = { color: "#000000", name: "Unknown" };

    return (classProperties[classId] || defaultProperties)[property];
}

// Function to create legend items dynamically
function createLegend() {
    // Input/Output: None

    const legendContainer = document.getElementById("class-legend");

    // Loop through class IDs and create legend items
    for (let classId = 1; classId <= 8; classId++) {
        const color = getClassProperty(classId, "color");
        const className = getClassProperty(classId, "name");
        const legendItem = document.createElement("div");
        legendItem.className = "legend-item";
        legendItem.innerHTML = `<span class="legend-color" style="background-color: ${color};"></span> ${className}`;
        legendContainer.appendChild(legendItem);
    }
}

// Call the function to create the legend
createLegend();

function clonePolygon(originalPolygon) {
    // Input: originalPolygon - Original Google Maps Polygon object
    // Output: Cloned Google Maps Polygon object

    return new google.maps.Polygon({
        paths: originalPolygon.getPaths().getArray().map(path => path.getArray()),
        strokeColor: originalPolygon.strokeColor,
        strokeOpacity: originalPolygon.strokeOpacity,
        strokeWeight: originalPolygon.strokeWeight,
        fillColor: originalPolygon.fillColor,
        fillOpacity: originalPolygon.fillOpacity
    });
}

// called on Google map API load
function initMap() {
    // bounding box
    var topLeft = { lat: 11.371305048234298, lng: -3.8928232831763268 };
    var topRight = { lat: 11.371305048234298, lng: -3.42 };
    var bottomRight = { lat: 10.96, lng: -3.42 };
    var bottomLeft = { lat: 10.96, lng: -3.8928232831763268 };

    // calculate the center of the bounding box
    var centerCoordinates = {
        lat: (topLeft.lat + bottomRight.lat) / 2,
        lng: (topLeft.lng + bottomRight.lng) / 2
    };

    // load the map
    map = new google.maps.Map(document.getElementById('map-container'), {
        zoom: 12,
        center: centerCoordinates
    });

    // create squared bounding box
    var boundingBox = new google.maps.Polygon({
        map: map,
        paths: [topLeft, topRight, bottomRight, bottomLeft],
        strokeColor: '#FF0000', // rouge
        strokeOpacity: 1,
        strokeWeight: 2,
        fillColor: 'transparent',
        fillOpacity: 0.2
    });

    let heatmapData = [];

    // start by fetching the input file
    fetch('./sources/input_file.json')
        .then(response => response.json())
        .then(data => {
            // Loop through the data and create polygons for each area
            data.forEach((areaData) => {
                const polygonCoordinates = areaData.perimeter_pixel_coordinates.map(coord => ({
                    lat: parseFloat(coord.latitude),
                    lng: parseFloat(coord.longitude),
                    intensity: parseFloat(coord.intensity)
                }));

                const areaColor = getClassProperty(areaData.class_id, "color");

                // Create a polygon with ordered coordinates
                const polygon = new google.maps.Polygon({
                    paths: polygonCoordinates,
                    map: map,
                    strokeColor: areaColor,
                    strokeOpacity: 1,
                    strokeWeight: 2,
                    fillColor: areaColor,
                    fillOpacity: 0.2
                });

                // Add click event listener to the polygon
                polygon.addListener("click", () => {
                    const infoContent = `
                    <h3>Class of the segment: ${getClassProperty(areaData.class_id, "name")}</h3>
                    `;
                    document.getElementById("info-content").innerHTML = infoContent; // display class name on right side panel
                    document.getElementById('info-panel').style.transform = "translateX(0)"; // display right side panel
                    document.getElementById('info-panel-arrow').style.display = "block"; // show the arrow

                    clickedPolygon = polygon;
                    clickedPixelMarkers = [];

                    // get clicked segment coordinates
                    const polygonCoordinates = areaData.perimeter_pixel_coordinates.map(coord => ({
                        lat: parseFloat(coord.latitude),
                        lng: parseFloat(coord.longitude),
                        intensity: parseFloat(coord.intensity)
                    }));

                    // Loop through pixel_coordinates and display them on the map with dynamic alpha values
                    areaData.pixel_coordinates.forEach((pixelCoord, index) => {
                        const alpha = areaData.alphas[index];
                        const pixelMarker = new google.maps.Marker({
                            position: new google.maps.LatLng(pixelCoord.latitude, pixelCoord.longitude),
                            map: map,
                            icon: {
                                path: google.maps.SymbolPath.CIRCLE,
                                fillColor: getColorFromAlpha(alpha),
                                fillOpacity: 1,
                                strokeWeight: 0,
                                scale: 5
                            }
                        });
                        clickedPixelMarkers.push(pixelMarker);
                    });


                    // calculate geometric center of the clicked segment
                    const centerLat = polygonCoordinates.reduce((sum, coord) => sum + coord.lat, 0) / polygonCoordinates.length;
                    const centerLng = polygonCoordinates.reduce((sum, coord) => sum + coord.lng, 0) / polygonCoordinates.length;

                    // Create a new map in the panel centered on the clicked area
                    const mapOptions = {
                        center: new google.maps.LatLng(centerLat, centerLng),
                        zoom: 20,
                        mapTypeId: google.maps.MapTypeId.SATELLITE
                    };
                    
                    mapInPanel = new google.maps.Map(document.getElementById("map-in-panel"), mapOptions);
                    
                    // Create a clone of the clicked polygon
                    const clonedPolygon = clonePolygon(polygon);

                    // Add the cloned polygon to the right side panel map
                    clonedPolygon.setMap(mapInPanel);

                    // Add the saved pixel markers to the existing map
                    clickedPixelMarkers.forEach(marker => {
                        marker.setMap(mapInPanel);
                    });

                    // Add console.log for debugging
                    console.log('polygonCoordinates V1828:', polygonCoordinates);

                    // Clear the previous weights
                    document.getElementById("legend").innerHTML = '';

                    // Calculate the weights for the two categories
                    const redWeight = Math.max(...areaData.alphas.map(alpha => parseFloat(alpha)));
                    const blueWeight = 1 - redWeight;
                    var red_color = "#f1c40f";
                    var blue_color = "#2c3e50";

                    if (BinaryClicked) {
                        red_color = 'red';
                        blue_color = 'blue';
                    } else {
                        // Convert alpha to a value between 0 and 255 for RGB
                        let alphaScaled_red = Math.floor(redWeight * 255);
                        let alphaScaled_blue = Math.floor(blueWeight * 255);
                        red_color = `rgb(${255 - alphaScaled_red}, 0, ${alphaScaled_red})`;
                        blue_color = `rgb(${255 - alphaScaled_blue}, 0, ${alphaScaled_blue})`;
                    }

                    // Update the #legend-content with the weights and colored boxes
                    const legendContent = `
                        <h4>Attention Weight</h4>
                        <div id="alphalegend">
                            <span class="legend-color-box" style="background-color: ${red_color};"></span>
                            <span>${redWeight.toFixed(2)}</span>
                            <span class="legend-color-box" style="background-color: ${blue_color};"></span>
                            <span>${blueWeight.toFixed(2)}</span>
                        </div>
                    `;
                    document.getElementById("legend").innerHTML = legendContent;
                });
                // Add the created polygon to the array
                allPolygons.push(polygon);

                // Loop through pixel_coordinates and add each pixel's location and alpha value to heatmapData
                areaData.pixel_coordinates.forEach((pixelCoord, index) => {
                    const alpha = areaData.alphas[index];
                    const location = new google.maps.LatLng(pixelCoord.latitude, pixelCoord.longitude);
                    heatmapData.push({location: location, weight: alpha});
                });
            });
            heatmap = new google.maps.visualization.HeatmapLayer({
                data: heatmapData,
                maxIntensity: 0.51,
                minIntensity: 0.49,
                //dissipating: false,
                map: map,
                radius: 10
            });
            
        })
        .catch(error => {
            console.error('Error fetching and concatenating data:', error);
        });
}