# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 2026-11-06

**Runtime:** máy cục bộ (CPU, không GPU), chạy lại đúng mã các ô của notebook `day1_understand_labels.ipynb` bằng script Python

**Python / PyTorch / Ultralytics:** Python 3.10.10 / PyTorch 2.7.1+cpu / Ultralytics 8.3.168

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt` (đã kiểm tra SHA-256 đúng như notebook)

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (không đổi)

**Thay đổi so với notebook nguồn:** Có, chỉ ở môi trường chạy: (1) không chạy trên Colab mà chạy cục bộ trên CPU; (2) dùng Ultralytics 8.3.168 đã cài sẵn thay vì ghim 8.4.145 (bỏ dòng `%pip install` và phép `assert` kiểm tra phiên bản), nên trường `ultralytics_version` trong JSON là `8.3.168`; (3) đổi backend matplotlib sang `Agg` và bỏ `plt.show()`. Checkpoint, ảnh, threshold và logic tạo JSON/PNG giữ nguyên; checksum model và ảnh đều PASS.

## 1. Phân loại ảnh

Dùng ảnh `traffic`, `rank = 1`.

1. `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh**. Record không có hộp hay đa giác, chỉ có một lớp cùng model score cho toàn ảnh; các hạng 2–5 (minibus, police_van, recreational_vehicle, streetcar) cũng là lớp xếp hạng cho cùng một ảnh đó, không phải các vật thể khác nhau.
3. Class list **đi kèm taxonomy/checkpoint** (ImageNet-1K, 1000 lớp) do người thiết kế dữ liệu huấn luyện định nghĩa. Model chỉ chọn trong danh sách có sẵn đó, không tự nghĩ ra lớp mới.
4. `class_id` chỉ có nghĩa trong một taxonomy cụ thể (id 468 ở ImageNet-1K khác id 468 ở COCO); `class_name` giúp người đọc hiểu; `taxonomy_name` cho biết id/tên thuộc bộ lớp nào. Thiếu một trong ba thì dễ nhầm lẫn hoặc không đối chiếu lại được khi đổi taxonomy.
5. Guideline phải nêu rõ **tiêu chí chọn một lớp**: lấy chủ thể chính (vật/chủ đề chiếm diện tích lớn nhất hoặc nổi bật nhất) hay theo bối cảnh chung, cách xử lý khi có nhiều chủ thể ngang nhau, và khi nào gán "không rõ/khác" hoặc escalate.
6. `score` chỉ là độ tin cậy của model, dùng để xếp hạng/lọc; model có thể tự tin nhưng sai. Ground truth phải do người gán theo guideline và được reviewer kiểm tra. Ví dụ ở đây `cab` (0.51) là lớp gần đúng nhất trong ImageNet cho cảnh giao thông, nhưng score chỉ hơn 0.5 và không chứng minh nhãn đúng.

## 2. Phát hiện vật thể

Dùng ảnh `kitchen`.

1. `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Góc trên-trái ở pixel (385.33, 69.24), góc dưới-phải ở pixel (498.92, 348.92), gốc (0,0) ở góc trên-trái ảnh, x tăng sang phải, y tăng xuống dưới. Hộp rộng 498.92 − 385.33 = 113.58 px và cao 348.92 − 69.24 = 279.68 px, bao quanh người đứng quay lưng ở bên phải ảnh.
3. Ngưỡng 0.20: **17** prediction; ngưỡng 0.35 (dùng cho JSON): **11**; ngưỡng 0.60: **6**.
4. Hạ ngưỡng thì có thêm nhiều hộp score thấp (ở 0.20 xuất hiện thêm spoon, potted plant, dining table, bottle). Reviewer phải xem xét thêm từng hộp: loại hộp thừa/sai lớp, kiểm tra hộp lệch, và đối chiếu xem còn vật bị bỏ sót hay không, nên khối lượng QC tăng và dễ lẫn prediction sai vào dữ liệu.
5. Quy tắc box: hộp phải bao sát toàn bộ phần nhìn thấy của vật thể (cạnh hộp chạm sát mép vật, không chừa khoảng trống lớn và không cắt vào vật), mỗi vật thể một hộp riêng, và gán đúng một lớp theo taxonomy.
6. Ở góc trái ảnh, `person` score 0.61 chỉ là bàn tay/cánh tay bị cắt ở mép ảnh, còn `oven` bên trái (0.69) thực chất là bếp lò/bếp nướng bị bàn che một phần. Các vật nhỏ như `bowl` (0.46) nằm xa và chỉ thấy một phần. Với quy tắc chung như "vật bị che/cắt mép thì vẽ hộp quanh phần nhìn thấy" tôi sẽ viết vào guideline vì lặp lại ở nhiều ảnh; còn trường hợp mơ hồ về **lớp** (bếp có tính là `oven` của COCO không, bàn tay có tính là `person` không) thì tôi **escalate lên Lab Coach** vì ranh giới taxonomy là quyết định của người viết guideline, annotator không tự bịa.

## 3. Phân đoạn theo từng đối tượng

Dùng ảnh `kitchen`.

1. `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; `polygon_point_count`=335 ; 4 điểm đầu của `polygon_xy`=[[443.0, 67.5], [443.0, 70.5], [441.0, 72.5], [439.0, 72.5]]
2. Hộp chỉ cho biết vùng chữ nhật chứa vật. Polygon bám theo biên thật: tách phần vật thể khỏi nền nằm trong hộp (ví dụ khoảng trống giữa hai chân, nền quanh tay người), cho biết hình dạng và diện tích thực, nên dùng được để tính vùng chiếm chỗ và tách vật khi hai hộp chồng nhau.
3. `instance_id` (dạng `kitchen-001`) chỉ để phân biệt từng instance trong file output của bài lab, ví dụ hai `bowl` khác nhau có hai id khác nhau. Nó không phải `class_id` (không nói vật thuộc lớp nào, nhiều instance cùng lớp) và không phải tracking ID (không theo dõi cùng một vật qua nhiều khung hình/ảnh; chạy lại hay đổi ảnh thì id không có ý nghĩa nối tiếp).
4. Quy tắc vẽ biên mask: đặt các điểm polygon sát theo đường biên thật của phần vật nhìn thấy, không lẫn nền, và mỗi instance một polygon riêng khép kín; chỗ bị che chỉ vẽ phần nhìn thấy.
5. Ở bàn trong ảnh `kitchen`, các `bowl` (kitchen-002, 003, 006) nằm sát nhau và chồng/chạm nhau cùng với đồ trên bàn nên biên giữa chúng mờ, và `dining table` (kitchen-005, 560 điểm) có biên bị đồ vật che nhiều chỗ. Quy tắc chung về chỗ chạm/che (vẽ phần nhìn thấy, không chồng mask) tôi dùng **guideline**; nếu không phân biệt được vật nào là vật nào (ví dụ chồng bát lẫn với đĩa/bánh) thì **escalation** cho Lab Coach thay vì đoán.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một lớp (class_id + class_name + taxonomy) cho cả ảnh | Ảnh `traffic` có nhiều xe nhưng top-1 `cab` chỉ 0.51, các hạng 2–5 (minibus, police_van, streetcar…) cũng hợp lý; lớp ImageNet không khớp hoàn toàn cảnh đường phố COCO | Chọn một lớp theo tiêu chí chủ thể chính trong guideline; ca mơ hồ thì escalate | Đúng lớp, đúng taxonomy, áp dụng nhất quán tiêu chí chọn lớp giữa các ảnh |
| Phát hiện vật thể | Một hộp `[x_min, y_min, x_max, y_max]` pixel + một lớp cho mỗi vật thể | Ảnh `kitchen`: `person` 0.61 chỉ là bàn tay cắt mép ảnh; `oven` trái là bếp bị che; nhiều `bowl`/`cup` nhỏ chồng nhau; hạ ngưỡng 0.20 xuất hiện thêm 6 hộp | Vẽ hộp sát phần nhìn thấy, gán đúng lớp, đủ mọi vật thuộc taxonomy | Hộp lệch/lỏng, sai lớp, vật bị bỏ sót, hộp thừa; kiểm tra hộp nằm trong ảnh |
| Instance segmentation | Một polygon (danh sách điểm pixel) + một lớp + một instance cho mỗi vật | Ảnh `kitchen`: các `bowl` sát nhau và `dining table` bị che có biên mờ; polygon của `person` có 335 điểm cần khép kín đúng biên | Vẽ biên sát vật, tách từng instance riêng kể cả cùng lớp; ca chồng/mờ theo guideline hoặc escalate | Biên mask bám đúng vật, không tràn sang vật khác, instance không bị gộp hoặc tách sai, điểm polygon trong ảnh |

## 5. An toàn dữ liệu

1. Chỉ dùng ba ảnh COCO công khai có sẵn (kèm attribution CC BY 2.0) và không đưa họ tên, MSSV, ảnh cá nhân hay dữ liệu nội bộ/khách hàng vào báo cáo, output hay GitHub công khai.
2. Nếu gặp ảnh hoặc dữ liệu ngoài phạm vi (có khuôn mặt, biển số, dữ liệu cá nhân/nội bộ), tôi dừng lại và báo cho Lab Coach (rồi GV/Lab Lead nếu cần), không tiếp tục tải lên hay xử lý.

## 6. Checklist trước khi nộp link repo

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
