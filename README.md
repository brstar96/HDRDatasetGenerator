# HDRDatasetGenerator
Panorama HDRI 이미지로부터 Perspective projected HDRIi 이미지를 생성하기 위한 코드입니다.  <br>

## Basic Usage
- HDRI panorama2perspective 변환기 (경로: `./panorama2perspective_hdri.py`)
    ```python
   python panorama2perspective_hdri.py --hdri_path <path> --save_path <path> --w_pers <int> --h_pers <int> --fov <float> --skew <float> --fix_roll <bool> --fix_pitch <bool> --fix_yaw <bool> --sample_mode <str> --seed <int> --viewpoint_count <int> --patch_count <int> --patch_res <int> --num_proc <int> --change_pname <bool>
   e.g.
   python panorama2perspective_hdri.py --hdri_path /home/datasets/HDRIHaven_subset --save_path /home/project_src/debug/HDRIDataset_debug --w_pers 3840 --h_pers 2160 --fov 90.0 --skew 0.0 --fix_roll False --fix_pitch True --fix_yaw False --sample_mode patchonly --seed 2021 --viewpoint_count 6 --patch_count 5 --patch_res 512 --num_proc 64 --change_pname True
    ```

    - 등장방형 파노라마 HDR 이미지로부터 지정된 해상도를 갖는 Perspective 이미지 또는 이미지 패치를 랜덤 Crop해 저장합니다. <br>
      - `hdri_path`: 입력 HDR 이미지 경로를 지정합니다. 
      - `save_path`: 자른 HDR 이미지 패치가 저장될 경로를 지정합니다. 
      - `w_pers`: 잘린 HDR 이미지의 가로 해상도 길이를 지정합니다. 
      - `h_pers`: 잘린 HDR 이미지의 세로 해상도 길이를 지정합니다. 
      - `fov`: 파노라마 이미지에서 Perspective 이미지를 캡쳐할 때 사용하는 카메라에 적용할 시야각 각도를 조정합니다. 
      - `skew`: 파노라마 이미지에서 Perspective 이미지를 캡쳐할 때 사용하는 카메라의 skew intrinsic parameter를 조정합니다. (기울기)
      - `fix_roll`: 파노라마 이미지에서 Perspective 이미지를 랜덤 캡쳐할 때 roll 축으로의 회전을 고정시킬지 말지 결정합니다. 
      - `fix_pitch`: 파노라마 이미지에서 Perspective 이미지를 랜덤 캡쳐할 때 pitch 축으로의 회전을 고정시킬지 말지 결정합니다. 
      - `fix_yaw`: 파노라마 이미지에서 Perspective 이미지를 랜덤 캡쳐할 때 yaw 축으로의 회전을 고정시킬지 말지 결정합니다. 
      - `sample_mode`: 파노라마 HDR 이미지를 어떤 형식으로 저장할지 결정합니다. 
        - `personly`: 파노라마 HDR 이미지에서 `w_pers`와 `h_pers` 해상도를 갖는 Perspective 이미지만 랜덤 크롭합니다. 
        - `patchonly`: 파노라마 이미지에서 `w_pers`와 `h_pers` 해상도를 갖는 Perspective 이미지를 랜덤 크롭한 후 `patch_count` 개수만큼의 `patch_res`해상도 패치를 잘라내 저장합니다. 
        - <i>(개발중) `both`: 파노라마 HDR 이미지에서 `w_pers`와 `h_pers` 해상도를 갖는 Perspective 이미지를 랜덤 크롭한 결과와, 랜덤 크롭된 Perspective 이미지로부터 `patch_count` 개수만큼의 `patch_res`해상도 패치를 잘라낸 결과를 모두 저장합니다. </i>
        - `jsononly`: 랜덤 샘플링된 Perspective 이미지 좌표와, `patch_count` 개수만큼 랜덤 샘플링된 패치 이미지의 좌표만 `*.json`포맷으로 저장합니다. 
      - `seed`: 랜덤시드 값을 지정합니다. 
      - `viewpoint_count`: 하나의 파노라마 HDR 이미지에서 랜덤 샘플링할 뷰포인트 좌표 개수를 지정합니다. `6`을 지정하는 경우, 하나의 파노라마 HDR 이미지에서 6장의 Perspective 이미지가 랜덤 샘플링됩니다. 
      - `patch_count`: 캡쳐된 Perspective 이미지에서 랜덤 샘플링할 패치 HDR 이미지 개수를 지정합니다. `5`를 지정하는 경우, 캡쳐된 Perspective HDR 이미지로부터 5장의 HDR 이미지 패치가 랜덤 샘플링됩니다. 
      - `patch_res`: 샘플링될 HDR 이미지 패치의 해상도를 결정합니다. 
      - `num_proc`: 전처리 작업에 사용될 프로세스 개수를 지정합니다. `64`로 지정하는 경우, 64개의 프로세스가 독립적으로 전처리 작업을 수행합니다. 
      - `change_pname`: 전처리 작업을 수행하는 동안 프로세스 이름을 진행도에 따라 수정할지 여부를 결정합니다. 

## Requirements
- Python (>=3.6)
- Pytorch

## Release Note
- Inverse Tone Mapping Utils
  - 2021.04.15 version [15c42f9]: 
    - 멀티프로세싱 기반 등장방형 HDRI 이미지 전처리 코드 업데이트 (*.hdr Equirectangular HDRI to perspective LDR *.png image)
    - 주요 지원 기능: 
      - 360도 등장방형 *.hdr 파일을 읽고 카메라 앵글과 패치 좌표를 랜덤 샘플링해 *.png LDR 이미지로 저장
      - 랜덤 샘플링된 카메라 앵글과 패치 좌표를 8:1:1로 분할
      - 모델이 출력한 Inverse Tone Mapped HDR 이미지와 정답 HDR 이미지 간 PSNR, SSIM 수치를 일괄 계산

### TODO:
- [x] Equilib 라이브러리로 등장방형 HDRI로부터 Perspective Projected HDRI 잘라내기
- [ ] [HDRI Haven](https://hdrihaven.com/) 크롤링

## Acknowledgements:
- [Equilib](https://github.com/haruishi43/equilib)
 