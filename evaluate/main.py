import os
from RAFT.scoring import raft_score


video_folder = r'/share0/dreamyou070/dreamyou070/VideoDistillation/experiment/2_1_down1_mid_up_02_webvid_distill_loss_1_feature_1_lr_scale_3/samples'
files = os.listdir(video_folder)
teacher_scores = []
student_scores = {}
for file in files:
    if file.endswith('.mp4'):
        name = file.split('.')[0]
        # [1] teacher file

        if 'teacher' in name:
            print(f'teacher name = {name}')
            video_path = os.path.join(video_folder, file)
            score = raft_score(video_path)
            teacher_scores.append(score)
        else :
            epoch = int(name.split('_')[-2])
            video_path = os.path.join(video_folder, file)
            score = raft_score(video_path)
            if epoch not in student_scores:
                student_scores[epoch] = []
            student_scores[epoch].append(score)

for epoch in student_scores.keys():
    student_scores[epoch] = sum(student_scores[epoch])/len(student_scores[epoch])
teacher_score = sum(teacher_scores)/len(teacher_scores)

print(f'teacher_score = {teacher_score}')
print(f'student_scores = {student_scores}')