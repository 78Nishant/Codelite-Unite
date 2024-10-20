from flask_cors import CORS
from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import pickle
from flask_pymongo import PyMongo
from dotenv import load_dotenv
import os
from pymongo import MongoClient
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


load_dotenv()



# loading models
Jmodel = pickle.load(open('jmodel.pkl', 'rb'))
Wmodel = pickle.load(open('wmodel.pkl', 'rb'))
Cmodel = pickle.load(open('cmodel.pkl', 'rb'))
Smodel = pickle.load(open('smodel.pkl', 'rb'))
Bmodel = pickle.load(open('bmodel.pkl', 'rb'))
Comodel = pickle.load(open('comodel.pkl', 'rb'))
Armodel = pickle.load(open('armodel.pkl', 'rb'))
Bamodel = pickle.load(open('bamodel.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

app = Flask(__name__, static_folder='../frontend/build', static_url_path='/')
CORS(app, resources={r"/api/*": {"origins": "*"}})
MONGO_URL = os.getenv("MONGO_URL")

app.config["MONGO_URI"] = MONGO_URL
mongo = PyMongo(app)

collection = mongo.db.crop_statistics


# Constants
DISTRICTS = ['Solapur', 'Nanded', 'Buldhana', 'Amravati', 'Sambhajinagar']
COMMODITIES = ['Jowar', 'Bajara', 'Cotton', 'Sugarcane', 'Wheat']


@app.route('/api/hello', methods=['GET'])
def hello():
    return jsonify({"message": "Hello from Flask! to darshan"}), 200


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

# @app.route('/api/crop-statistics', methods=['GET'])
# def get_crop_statistics():
#     try:
#         # Get all data
#         cursor = collection.find({})
#         data = list(cursor)
        
#         # Initialize matrices
#         crop_per_dist = [[0 for _ in range(5)] for _ in range(5)]
#         crop_state = [0] * 5
        
#         # Calculate distribution
#         for item in data:
#             try:
#                 commodity_index = COMMODITIES.index(item.get('commodity'))
#                 dist_index = DISTRICTS.index(item.get('district'))
#                 crop_per_dist[dist_index][commodity_index] += 1
#             except (ValueError, TypeError):
#                 continue
        
#         # Calculate state totals
#         for i in range(5):
#             crop_state[i] = sum(row[i] for row in crop_per_dist)
        
#         # Prepare detailed statistics
#         district_statistics = {
#             district: {
#                 commodity: crop_per_dist[d_idx][c_idx]
#                 for c_idx, commodity in enumerate(COMMODITIES)
#             }
#             for d_idx, district in enumerate(DISTRICTS)
#         }
        
#         response_data = {
#             "state_level_statistics": {
#                 commodity: count
#                 for commodity, count in zip(COMMODITIES, crop_state)
#             },
#             "district_level_statistics": district_statistics,
#             "raw_data": {
#                 "crop_state_data": crop_state,
#                 "crop_per_district": crop_per_dist
#             },
#             "metadata": {
#                 "timestamp": datetime.now().isoformat(),
#                 "total_records": len(data),
#                 "districts": DISTRICTS,
#                 "commodities": COMMODITIES
#             }
#         }
        
#         return jsonify(response_data), 200
        
#     except Exception as e:
#         logger.error(f"Error in crop statistics: {e}")
#         return jsonify({
#             "error": str(e),
#             "success": False
#         }), 500

@app.route('/api/crop_statistics', methods=['POST', 'GET'])
def crop_statistics():
    try:
        distlist = ['Solapur', 'Nanded', 'Buldhana', 'Amravati', 'Sambhajinagar']
        commoditylist = ['Jowar', 'Bajara', 'Cotton', 'Sugarcane', 'Wheat']

        cursor = collection.find({})
        data = list(cursor)

        crop_per_dist = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]

        for dataele in data:
            commodityindex = commoditylist.index(dataele.get('commodity'))
            distindex = distlist.index(dataele.get('district'))
            crop_per_dist[distindex][commodityindex] += 1

        crop_state = [0, 0, 0, 0, 0]
        for i, ele in enumerate(crop_per_dist):
            crop_state[i] = sum(ele)

        response_data = {
            "crop_state_data": crop_state,
            "crop_per_district": crop_per_dist,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_records": len(data),
                "districts": distlist,
                "commodities": commoditylist
            }
        }

        return jsonify(response_data), 200

    except Exception as e:
        logger.error(f"Error in crop statistics: {e}")
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@app.route('/api/commodity/<commodity_name>', methods=['GET'])
def get_commodity_statistics(commodity_name):
    try:
            
        # Get commodity-specific data
        cursor = collection.find({'commodity': commodity_name})
        data = list(cursor)
        
        # Calculate frequency distribution
        frequency_count = [0] * len(DISTRICTS)
        for dist in data:
            try:
                dist_index = DISTRICTS.index(dist.get('district'))
                frequency_count[dist_index] += 1
            except ValueError:
                continue
        
        # Prepare detailed response
        response_data = {
            "commodity": commodity_name,
            "statistics": {
                "total_count": sum(frequency_count),
                "district_distribution": {
                    district: count
                    for district, count in zip(DISTRICTS, frequency_count)
                },
                "raw_frequency_count": frequency_count
            },
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "districts": DISTRICTS
            }
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Error in commodity statistics: {e}")
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

# Optional: Get all commodities data at once
@app.route('/api/commodities', methods=['GET'])
def get_all_commodities_statistics():
    try:
        all_commodities_data = {}
        
        for commodity in COMMODITIES:
            cursor = collection.find({'commodity': commodity})
            data = list(cursor)
            
            frequency_count = [0] * len(DISTRICTS)
            for dist in data:
                try:
                    dist_index = DISTRICTS.index(dist.get('district'))
                    frequency_count[dist_index] += 1
                except ValueError:
                    continue
                    
            all_commodities_data[commodity] = {
                "total_count": sum(frequency_count),
                "district_distribution": {
                    district: count
                    for district, count in zip(DISTRICTS, frequency_count)
                }
            }
        
        response_data = {
            "commodities": all_commodities_data,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "districts": DISTRICTS
            }
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Error in all commodities statistics: {e}")
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict_price():
    try:
        # Get data from request
        data = request.get_json()
        commoditytype = data['commodityname']
        month = data['month']
        year = data['year']
        next_year = int(year) + 1
        average_rain_fall = data['average_rain_fall']

        # Initialize arrays
        avg_price_year = []
        msp_year = []
        msp_next_year = []
        avg_price_next_year = []
        months_labels = ["Jan", "Feb", "March", "April", "May", "June", "July", "Aug", "Sept", "Oct", "Nov", "Dec"]
        month_count = 1
        year_count = 2021

        rainfall2024 = [1, 2, 3, 1, 8, 673, 1318, 779, 408, 106, 44, 8]
        rainfall2023 = [90, 7.2, 29.9, 41.4, 67.6, 148.6, 315.9, 162.7, 190.3, 50.8, 9.3, 8]

        # Prepare features for prediction
        features = np.array([[month, year, average_rain_fall]], dtype=object)
        transformed_features = preprocessor.transform(features)

        # Dictionary mapping commodity types to their price factors
        commodity_configs = {
            "Jowar": {"model": Jmodel, "min_factor": 1550, "max_factor": 2970},
            "Wheat": {"model": Wmodel, "min_factor": 1350, "max_factor": 2125},
            "Cotton": {"model": Cmodel, "min_factor": 3600, "max_factor": 6080},
            "Sugarcane": {"model": Smodel, "min_factor": 2250, "max_factor": 2775},
            "Bajara": {"model": Bmodel, "min_factor": 1175, "max_factor": 2350},
            "Copra": {"model": Comodel, "min_factor": 4000, "max_factor": 6500},
            "Arhar": {"model": Armodel, "min_factor": 4940, "max_factor": 6500},
            "Barley": {"model": Bamodel, "min_factor": 1300, "max_factor": 2200}  
        }

        if commoditytype not in commodity_configs:
            return jsonify({"error": "Invalid commodity type"}), 400

        config = commodity_configs[commoditytype]
        model = config["model"]
        min_factor = config["min_factor"]
        max_factor = config["max_factor"]

        # Get main prediction
        prediction = model.predict(transformed_features).reshape(1, -1)
        predicted_value = round(prediction[0][0], 3)
        min_value = round((predicted_value * min_factor) / 100, 2)
        max_value = round((predicted_value * max_factor) / 100, 2)
        avg_value = round((min_value + max_value) / 2, 2)

        # Calculate yearly predictions
        for rainfall in rainfall2023:
            features = np.array([[month_count, year_count, rainfall]], dtype=object)
            transformed_features = preprocessor.transform(features)
            prediction = model.predict(transformed_features).reshape(1, -1)
            predicted_value = round(prediction[0][0], 3)
            
            msp_year.append(round((predicted_value * max_factor) / 100, 2))
            avg_price_year.append(round((predicted_value * min_factor) / 100, 2))
            month_count += 1

        # Calculate next year predictions
        for rainfall in rainfall2024:
            features = np.array([[month_count, year_count, rainfall]], dtype=object)
            transformed_features = preprocessor.transform(features)
            prediction = model.predict(transformed_features).reshape(1, -1)
            predicted_value = round(prediction[0][0], 3)
            
            msp_next_year.append(round((predicted_value * max_factor) / 100, 2))
            avg_price_next_year.append(round((predicted_value * min_factor) / 100, 2))

        # Calculate statistics
        max_msp_year = max(msp_year)
        max_avg_price_year = max(avg_price_year)
        min_msp_year = min(msp_year)
        min_avg_price_year = min(avg_price_year)
        gold_month_index = msp_year.index(max_msp_year) + 1
        silver_month_index = msp_year.index(min_msp_year) + 1

        response_data = {
            "prediction": predicted_value,
            "price_range": {
                "min_value": min_value,
                "max_value": max_value,
                "avg_value": avg_value
            },
            "time_period": {
                "year": year,
                "next_year": next_year,
                "month": month
            },
            "yearly_statistics": {
                "max_price": {
                    "msp": max_msp_year,
                    "average": max_avg_price_year
                },
                "min_price": {
                    "msp": min_msp_year,
                    "average": min_avg_price_year
                },
                "best_month_index": gold_month_index,
                "worst_month_index": silver_month_index
            },
            "monthly_data": {
                "months": months_labels,
                "current_year": {
                    "msp": msp_year,
                    "average_price": avg_price_year
                },
                "next_year": {
                    "msp": msp_next_year,
                    "average_price": avg_price_next_year
                }
            }
        }

        return jsonify(response_data), 200

    except KeyError as e:
        return jsonify({"error": f"Missing required field: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/submit_crop_data', methods=['POST'])
def submit_crop_data():
    data = request.json
    collection.insert_one(data)
    return jsonify({"message": "Data submitted successfully"}), 201

# @app.route('/api/predict', methods=['POST'])
# def predict():
#     data = request.json
#     commoditytype = data['commodityname']
#     month = int(data['month'])
#     year = int(data['year'])
#     average_rain_fall = float(data['average_rain_fall'])

#     features = np.array([[month, year, average_rain_fall]], dtype=object)
#     transformed_features = preprocessor.transform(features)

#     if commoditytype == "Jowar":
#         model = Jmodel
#         min_factor, max_factor = 1550, 2970
#     elif commoditytype == "Wheat":
#         model = Wmodel
#         min_factor, max_factor = 1350, 2125
#     elif commoditytype == "Cotton":
#         model = Cmodel
#         min_factor, max_factor = 3600, 6080
#     elif commoditytype == "Sugarcane":
#         model = Smodel
#         min_factor, max_factor = 2250, 2775
#     elif commoditytype == "Bajara":
#         model = Bmodel
#         min_factor, max_factor = 1175, 2350
#     else:
#         return jsonify({"error": "Invalid commodity type"}), 400

#     prediction = model.predict(transformed_features).reshape(1, -1)
#     predicted_value = round(prediction[0][0], 3)
#     min_value = round((predicted_value * min_factor) / 100, 2)
#     max_value = round((predicted_value * max_factor) / 100, 2)
#     avg_value = round((min_value + max_value) / 2, 2)

#     return jsonify({
#         "prediction": predicted_value,
#         "min_value": min_value,
#         "max_value": max_value,
#         "avg_value": avg_value,
#         "year": year,
#         "month": month,
#         "commodity": commoditytype
#     })

@app.route('/api/commodity/<string:commodity>', methods=['GET'])
def commodity_data(commodity):
    cursor = collection.find({'commodity': commodity})
    data = list(cursor)

    distlist = ['Solapur', 'Nanded', 'Buldhana', 'Amravati', 'Sambhajinagar']
    frequency_count = [0, 0, 0, 0, 0]

    for district in distlist:
        frequency_count[distlist.index(district)] = sum(1 for dist in data if dist.get('district') == district)

    return jsonify({
        "commodity": commodity,
        "frequency_data": frequency_count
    })

@app.route('/api/district/<district_name>', methods=['GET'])
def get_district_data(district_name):
    try:
        # Dictionary to handle district name mappings (if any)
        district_mappings = {
            "Amaravati": "Amravati",  # Handle potential spelling variations
            "Sambhajinar": "Sambhajinagar"  # Handle potential spelling variations
        }

        # Get the correct district name from mappings or use the provided name
        search_district = district_mappings.get(district_name, district_name)

        # Validate if it's a supported district
        supported_districts = [
            "Solapur", "Nanded", "Buldhana", 
            "Amravati", "Sambhajinagar"
        ]
        
        if search_district not in supported_districts:
            return jsonify({
                "error": "Invalid district name",
                "supported_districts": supported_districts
            }), 400

        # Query the database
        cursor = collection.find({'district': search_district})
        data = list(cursor)

        # Initialize crop frequency counting
        commodity_list = ['Jowar', 'Bajara', 'Cotton', 'Sugarcane', 'Wheat']
        crop_frequency = [0] * len(commodity_list)

        # Count crop frequencies
        for data_element in data:
            try:
                commodity_index = commodity_list.index(data_element.get('commodity'))
                crop_frequency[commodity_index] += 1
            except ValueError:
                # Handle case where commodity is not in our list
                continue

        # Prepare response data
        response_data = {
            "district": district_name,
            "crop_statistics": {
                "total_records": len(data),
                "crops": {
                    commodity: frequency 
                    for commodity, frequency in zip(commodity_list, crop_frequency)
                },
                "crop_frequency": crop_frequency,  # Keep the original array format if needed
                "commodity_list": commodity_list   # Include the list of commodities
            },
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
        }

        return jsonify(response_data), 200

    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

# Optional: Add an endpoint to get all districts' data
@app.route('/api/districts', methods=['GET'])
def get_all_districts_data():
    try:
        districts = ["Solapur", "Nanded", "Buldhana", "Amravati", "Sambhajinagar"]
        all_districts_data = {}

        for district in districts:
            cursor = collection.find({'district': district})
            data = list(cursor)
            
            commodity_list = ['Jowar', 'Bajara', 'Cotton', 'Sugarcane', 'Wheat']
            crop_frequency = [0] * len(commodity_list)

            for data_element in data:
                try:
                    commodity_index = commodity_list.index(data_element.get('commodity'))
                    crop_frequency[commodity_index] += 1
                except ValueError:
                    continue

            all_districts_data[district] = {
                "crops": {
                    commodity: frequency 
                    for commodity, frequency in zip(commodity_list, crop_frequency)
                },
                "crop_frequency": crop_frequency
            }

        response_data = {
            "districts": all_districts_data,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
        }

        return jsonify(response_data), 200

    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        }), 500
# @app.route('/api/district/<string:district>', methods=['GET'])
# def district_data(district):
#     cursor = collection.find({'district': district})
#     data = list(cursor)

#     commoditylist = ['Jowar', 'Bajara', 'Cotton', 'Sugarcane', 'Wheat']
#     crop_frequency = [0, 0, 0, 0, 0]

#     for dataele in data:
#         commodityindex = commoditylist.index(dataele.get('commodity'))
#         crop_frequency[commodityindex] += 1

#     return jsonify({
#         "district": district,
#         "crop_frequency": crop_frequency
#     })

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)