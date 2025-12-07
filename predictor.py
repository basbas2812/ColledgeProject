from flask import Flask, request, jsonify
import cv2
import joblib
import numpy as np
from skimage.feature import hog
import mysql.connector


target_names_map = {
    "มะเขือเทศ": [
        'Tomato___Bacterial_spot',
        'Tomato___Early_blight',
        'Tomato___Late_blight',
        'Tomato___Leaf_Mold',
        'Tomato___Septoria_leaf_spot',
        'Tomato___Spider_mites Two-spotted_spider_mite',
        'Tomato___Target_Spot',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
        'Tomato___Tomato_mosaic_virus',
        'Tomato___healthy'
    ],
    "ข้าวโพด": [
        'cercospora leaf spot gray leaf spot',
        'common rust',
        'healthy',
        'northern leaf blight'
    ],
    "มันสำปะหลัง": [
        'Bacterial Blight (CBB)',
        'Brown Streak Disease (CBSD)',
        'Green Mottle (CGM)',
        'Mosaic Disease (CMD)',
        'Healthy'
    ],
    "ทุเรียน": [
        'ALGAL_LEAF_SPOT',
        'ALLOCARIDARA_ATTACK',
        'HEALTHY_LEAF',
        'LEAF_BLIGHT',
        'PHOMOPSIS_LEAF_SPOT'
    ],
    "ข้าว": [
        'BrownSpot',
        'Healthy',
        'Hispa',
        'LeafBlast'
    ]
}

disease_mapping = {
    "Tomato___Bacterial_spot": {"id": "D0006", "name_th": "โรคจุดแบคทีเรีย"},
    "Tomato___Early_blight": {"id": "D0007", "name_th": "โรคใบไหม้ต้น"},
    "Tomato___Late_blight": {"id": "D0008", "name_th": "โรคใบไหม้ปลาย"},
    "Tomato___Leaf_Mold": {"id": "D0009", "name_th": "โรคราใบมะเขือเทศ"},
    "Tomato___Septoria_leaf_spot": {"id": "D0010", "name_th": "โรคจุดใบเซปโทเรีย"},
    "Tomato___Spider_mites Two-spotted_spider_mite": {"id": "D0011", "name_th": "ไรแดงสองจุด"},
    "Tomato___Target_Spot": {"id": "D0012", "name_th": "โรคจุดเป้า"},
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {"id": "D0013", "name_th": "ไวรัสใบหงิกเหลือง"},
    "Tomato___Tomato_mosaic_virus": {"id": "D0014", "name_th": "ไวรัสโมเสก"},
    "Tomato___healthy": {"id": None, "name_th": "ปกติ (ไม่พบโรค)"},
    "common rust": {"id": "D0016", "name_th": "โรคราสนิม"},
    "cercospora leaf spot gray leaf spot": {"id": "D0017", "name_th": "โรคใบจุดเซอร์โคสปอรา"},
    "northern leaf blight": {"id": "D0015", "name_th": "โรคใบไหม้แผลใหญ่"},
    "healthy": {"id": None, "name_th": "ไม่พบโรค"},
    "Bacterial Blight (CBB)": {"id": "D0021", "name_th": "โรคใบไหม้"},
    "Brown Streak Disease (CBSD)": {"id": "D0020", "name_th": "โรคแผลขีดสีน้ำตาล"},
    "Green Mottle (CGM)": {"id": "D0019", "name_th": "โรคใบด่างเขียว"},
    "Mosaic Disease (CMD)": {"id": "D0018", "name_th": "โรคใบด่างมันสำปะหลัง"},
    "Healthy": {"id": None, "name_th": "ไม่พบโรค"},
    "HEALTHY_LEAF": {"id": None, "name_th": "ไม่พบโรค"},
    "ALGAL_LEAF_SPOT": {"id": "D0022", "name_th": "โรคใบจุดสาหร่าย"},
    "ALLOCARIDARA_ATTACK": {"id": "D0023", "name_th": "เพลี้ยไก่แจ้ทุเรียน"},
    "LEAF_BLIGHT": {"id": "D0024", "name_th": "โรคใบไหม้"},
    "PHOMOPSIS_LEAF_SPOT": {"id": "D0025", "name_th": "โรคใบจุด"},
    "BrownSpot": {"id": "D0026", "name_th": "โรคใบจุดสีน้ำตาล"},
    "Hispa": {"id": "D0027", "name_th": "แมลงกรีดใบข้าว"},
    "LeafBlast": {"id": "D0028", "name_th": "โรคไหม้ในข้าว"},
}


app = Flask(__name__)

import mysql.connector

def get_disease_info(disease_id):
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="1234",
        database="myproject"
    )
    cursor = conn.cursor(dictionary=True)

    query = """
        SELECT d.diseaseId, d.diseaseName, d.symptoms,
               t.treatmentMethods,
               m.medicineName, m.dosage
        FROM disease d
        LEFT JOIN treatment t ON d.diseaseId = t.diseaseId
        LEFT JOIN medicine m ON d.diseaseId = m.diseaseId
        WHERE d.diseaseId = %s
    """
    cursor.execute(query, (disease_id,))
    results = cursor.fetchall()

    cursor.close()
    conn.close()

    return results



def img_segmentation(image):
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)

    lower_green = np.array([25,0,20])
    upper_green = np.array([100,255,255])
    healthy_mask = cv2.inRange(hsv_img, lower_green, upper_green)

    lower_brown = np.array([10,0,10])
    upper_brown = np.array([30,255,255])
    disease_mask = cv2.inRange(hsv_img, lower_brown, upper_brown)

    final_mask = healthy_mask + disease_mask
    final_result = cv2.bitwise_and(rgb_img, rgb_img, mask=final_mask)
    return final_result


import os
from werkzeug.utils import secure_filename

@app.route('/predict', methods=['POST'])
def predict():
    try:
        files = request.files.getlist('file')
        plant_type = request.form.get('plantType')

        if not files or len(files) == 0:
            return jsonify({"error": "กรุณาอัปโหลดไฟล์ภาพ"}), 400

        # ✅ ถ้าเลือก "อื่นๆ" หรือไม่ได้เลือก → ข้ามการ predict
        if not plant_type or plant_type == "orther":
            upload_dir = os.path.join("static", "uploads")
            os.makedirs(upload_dir, exist_ok=True)

            results = []
            for file in files:
                filename = secure_filename(file.filename)
                save_path = os.path.join(upload_dir, filename)
                file.save(save_path)

                results.append({
                    "ไฟล์": filename,
                    "โรค": "ไม่สามารถวิเคราะห์ได้",
                    "รายละเอียด": "กรุณาส่งต่อให้ผู้เชี่ยวชาญ",
                    "เปอร์เซ็นต์ความถูกต้อง": "ไม่ระบุ",
                    "image": f"uploads/{filename}"
                })

            return jsonify({
                "จำนวนภาพที่อัปโหลด": len(results),
                "ผลลัพธ์แต่ละภาพ": results,
                "สรุปผลรวม": "ไม่สามารถสรุปได้ (พืชไม่อยู่ในระบบ)"
            })

        model_map = {
            "มะเขือเทศ": "modelML/model_NB_Tomato.joblib",
            "ข้าว": "modelML/model_NB_rice.joblib",
            "ทุเรียน": "modelML/model_NB_durian.joblib",
            "ข้าวโพด": "modelML/model_NB_corn.joblib",
            "มันสำปะหลัง": "modelML/model_NB_cassava.joblib"
        }

        model_file = model_map.get(plant_type)
        if not model_file:
            return jsonify({"error": "ไม่พบโมเดลสำหรับพืชชนิดนี้"}), 400

        classifier = joblib.load(model_file)
        plant_target_names = target_names_map.get(plant_type, [])
        if not plant_target_names:
            return jsonify({"error": "ไม่พบ target names สำหรับพืชนี้"}), 400

        results = []
        total_confidence = 0
        upload_dir = os.path.join("static", "uploads")
        os.makedirs(upload_dir, exist_ok=True)

        for file in files:
            filename = secure_filename(file.filename)
            save_path = os.path.join(upload_dir, filename)
            file.save(save_path)

            image = cv2.imread(save_path)
            if image is None:
                results.append({
                    "ไฟล์": filename,
                    "ข้อผิดพลาด": "ไม่สามารถอ่านไฟล์ภาพได้",
                    "image": f"uploads/{filename}"
                })
                continue

            image = cv2.resize(image, (256, 256))
            segmented_img = img_segmentation(image)
            gray_image = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)

            features, _ = hog(
                gray_image, orientations=8,
                pixels_per_cell=(8, 8),
                cells_per_block=(1, 1), visualize=True
            )
            features = features.reshape(1, -1)

            if hasattr(classifier, "predict_proba"):
                proba = classifier.predict_proba(features)
                confidence = np.max(proba[0]) * 100
            else:
                confidence = 0.0
            total_confidence += confidence

            result = classifier.predict(features)
            try:
                predicted_label = plant_target_names[result[0]]
            except IndexError:
                results.append({
                    "ไฟล์": filename,
                    "ข้อผิดพลาด": "ผลลัพธ์โมเดลไม่ตรงกับ label ที่กำหนด",
                    "image": f"uploads/{filename}"
                })
                continue

            disease_info = disease_mapping.get(predicted_label)
            if not disease_info:
                # ✅ กรณีไม่พบโรคในระบบ → ส่งกลับเฉพาะชื่อ, เปอร์เซ็นต์, รูป
                results.append({
                    "ไฟล์": filename,
                    "โรค": "ไม่พบโรค",
                    "เปอร์เซ็นต์ความถูกต้อง": confidence,
                    "image": f"uploads/{filename}"
                })
                continue

            disease_id = disease_info["id"]
            disease_name_th = disease_info["name_th"]

            db_data = get_disease_info(disease_id) if disease_id else None
            if not db_data:
                # ✅ กรณีมีชื่อโรค แต่ยังไม่มีข้อมูลในฐานข้อมูล → ส่งเฉพาะชื่อ, เปอร์เซ็นต์, รูป
                results.append({
                    "ไฟล์": filename,
                    "โรค": disease_name_th,
                    "เปอร์เซ็นต์ความถูกต้อง": confidence,
                    "image": f"uploads/{filename}"
                })
                continue

            # ✅ กรณีพบโรคและมีข้อมูลครบ
            results.append({
                "ไฟล์": filename,
                "โรค": disease_name_th,
                "อาการ": db_data[0]["symptoms"],
                "วิธีรักษา": db_data[0]["treatmentMethods"],
                "ยา": db_data[0]["medicineName"],
                "ขนาดยา": db_data[0]["dosage"],
                "เปอร์เซ็นต์ความถูกต้อง": confidence,
                "image": f"uploads/{filename}"
            })

        avg_confidence = round(total_confidence / len(results), 2) if results else 0
        disease_counts = {}
        for r in results:
            if "โรค" in r:
                disease_name = r["โรค"]
                disease_counts[disease_name] = disease_counts.get(disease_name, 0) + 1

        total = sum(disease_counts.values())

        sorted_diseases = sorted(disease_counts.items(), key=lambda x: x[1], reverse=True)

        summary_parts = []
        for disease, count in sorted_diseases:
            percentage = round((count / total) * 100, 2)
            summary_parts.append(f"น่าจะเป็น {disease} ({percentage}%)")

        print(summary_parts)
        return jsonify({
            "จำนวนภาพที่วิเคราะห์": len(results),
            "ค่าเฉลี่ยความถูกต้อง": avg_confidence,
            "ผลลัพธ์แต่ละภาพ": results,
            "สรุปผลรวม": summary_parts    
        })


    except Exception as e:
        import traceback
        print("❌ ERROR:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    from waitress import serve
    app.run(host='0.0.0.0', port=5001, debug=True)
    #serve(app, host='0.0.0.0', port=5001)