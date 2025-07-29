"""
EUREKA: 대규모 언어 모델을 통한 인간 수준의 보상 설계

이 스크립트는 대규모 언어 모델(LLM)을 사용하여 강화학습 환경을 위한 보상 함수를
자동으로 생성하는 Eureka 프레임워크를 구현합니다. 이 알고리즘은 진화적 피드백
메커니즘을 통해 보상 함수 설계를 반복적으로 개선합니다.

논문: "Eureka: Human-Level Reward Design via Coding Large Language Models"
저자: Yecheng Jason Ma, William Liang, Guanzhi Wang, De-An Huang, Osbert Bastani,
      Dinesh Jayaraman, Yuke Zhu, Linxi Fan, Anima Anandkumar

핵심 아이디어는 LLM을 사용하여 코드로 보상 함수를 작성하고, 이를 RL 훈련에서
실행한 후, 성능 피드백을 제공하여 보상 설계를 반복적으로 개선하는 것입니다.
"""

# 다양한 유틸리티를 위한 표준 라이브러리 임포트
import hydra                    # YAML 파일을 사용한 설정 관리
import numpy as np              # 수치 계산 및 배열 연산
import json                     # LLM과의 대화 기록 저장/로드
import logging                  # 전체 프로세스에 대한 포괄적인 로깅
import matplotlib.pyplot as plt # 훈련 진행상황 및 성공률 그래프
import os                       # 파일 시스템 작업
import openai                   # OpenAI의 GPT 모델과 상호 작용
import re                       # 정규 표현식 패턴 매칭 (코드 추출용)
import subprocess               # 외부 프로세스 실행 (RL 훈련)
from pathlib import Path        # 현대적인 경로 처리
import shutil                   # 파일 복사 작업
import time                     # API 재시도 로직에서 지연 추가

# 커스텀 유틸리티 임포트
from utils.misc import *                                                    # 일반 유틸리티 함수
from utils.file_utils import find_files_with_substring, load_tensorboard_logs  # 파일 작업 및 텐서보드 로그 파싱
from utils.create_task import create_task                                  # 새로운 태스크를 위한 YAML 설정 파일 생성
from utils.extract_task_code import *                                      # 태스크 파일에서 코드를 추출하고 파싱하는 함수

# 전역 디렉토리 경로 - 프로젝트 구조를 정의
EUREKA_ROOT_DIR = os.getcwd()                                             # 현재 작업 디렉토리 (Eureka 프로젝트 루트)
ISAAC_ROOT_DIR = f"{EUREKA_ROOT_DIR}/../isaacgymenvs/isaacgymenvs"         # Isaac Gym 환경 경로


@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg):
    """
    자동화된 보상 함수 생성을 위한 Eureka 알고리즘을 구현하는 메인 함수.

    알고리즘은 다음 단계로 작동합니다:
    1. 설정: 구성 로드, API 키 설정, 환경 파일 준비
    2. 반복적 생성 루프:
       - LLM을 사용하여 보상 함수 후보 생성
       - 각 후보로 RL 훈련 실행
       - 성능 평가 및 피드백 제공
       - 다음 반복을 위한 최고 성능 후보 선택
    3. 최종 평가: 강건한 평가를 위해 최고 보상 함수를 여러 번 테스트

    Args:
        cfg: 모든 하이퍼파라미터와 설정을 포함하는 Hydra 구성 객체
    """

    # ================================================================================
    # 1단계: 초기화 및 설정
    # ================================================================================

    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {EUREKA_ROOT_DIR}")

    # 보안을 위해 외부 파일에서 OpenAI API 키 로드
    # API 키 파일은 프로젝트 루트에서 한 디렉토리 위에 위치해야 함
    api_key_path = os.path.abspath(
        os.path.join(EUREKA_ROOT_DIR, "../openai_api.txt")
    )
    try:
        with open(api_key_path, "r", encoding="utf-8") as f:
            openai.api_key = f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find API key file at {api_key_path}")

    # 주요 구성 매개변수 추출
    task = cfg.env.task                    # RL 태스크 이름 (예: "FrankaCabinet")
    task_description = cfg.env.description # 태스크가 달성해야 할 목표에 대한 인간이 읽을 수 있는 설명
    suffix = cfg.suffix                    # 생성된 태스크 파일에 추가할 접미사 (예: "Eureka")
    model = cfg.model                      # 사용할 LLM 모델 (예: "gpt-4", "gpt-3.5-turbo")

    logging.info(f"Using LLM: {model}")
    logging.info("Task: " + task)
    logging.info("Task description: " + task_description)

    # 환경 타입 결정 및 관련 파일 찾기
    # Eureka는 두 가지 타입의 환경을 지원: 'isaac' (Isaac Gym)과 'dexterity' (손 조작)
    env_name = cfg.env.env_name.lower()
    env_parent = 'isaac' if f'{env_name}.py' in os.listdir(f'{EUREKA_ROOT_DIR}/envs/isaac') else 'dexterity'

    # 기본 환경과 관찰 코드를 위한 파일 경로
    task_file = f'{EUREKA_ROOT_DIR}/envs/{env_parent}/{env_name}.py'           # 메인 환경 구현
    task_obs_file = f'{EUREKA_ROOT_DIR}/envs/{env_parent}/{env_name}_obs.py'   # 관찰 공간 정의

    # 참조용 관찰 파일 복사 및 코드를 문자열로 로드
    shutil.copy(task_obs_file, f"env_init_obs.py")
    task_code_string = file_to_string(task_file)           # 수정을 위해 환경 코드를 문자열로 로드
    task_obs_code_string = file_to_string(task_obs_file)   # LLM 컨텍스트를 위한 관찰 코드 로드

    # 생성된 환경 파일이 저장될 출력 경로 정의
    output_file = f"{ISAAC_ROOT_DIR}/tasks/{env_name}{suffix.lower()}.py"

    # ================================================================================
    # LLM 커뮤니케이션을 위한 프롬프트 템플릿 로드
    # ================================================================================

    # Eureka는 보상 함수 생성에서 LLM을 안내하기 위해 신중하게 작성된 프롬프트를 사용
    prompt_dir = f'{EUREKA_ROOT_DIR}/utils/prompts'

    # 핵심 프롬프트 구성 요소:
    initial_system = file_to_string(f'{prompt_dir}/initial_system.txt')           # LLM의 역할을 정의하는 시스템 메시지
    code_output_tip = file_to_string(f'{prompt_dir}/code_output_tip.txt')         # 코드 형식화 지침
    code_feedback = file_to_string(f'{prompt_dir}/code_feedback.txt')             # 코드 피드백 제공 템플릿
    initial_user = file_to_string(f'{prompt_dir}/initial_user.txt')               # 태스크 설명이 포함된 초기 사용자 프롬프트
    reward_signature = file_to_string(f'{prompt_dir}/reward_signature.txt')       # 보상을 위한 예상 함수 시그니처
    policy_feedback = file_to_string(f'{prompt_dir}/policy_feedback.txt')         # RL 성능 피드백 템플릿
    execution_error_feedback = file_to_string(f'{prompt_dir}/execution_error_feedback.txt')  # 오류 피드백 템플릿

    # LLM과의 초기 대화 구성
    # 시스템 메시지는 LLM의 역할을 정의하고 보상 함수 시그니처를 포함
    initial_system = initial_system.format(task_reward_signature_string=reward_signature) + code_output_tip

    # 사용자 메시지는 태스크와 관찰 공간에 대한 컨텍스트를 제공
    initial_user = initial_user.format(task_obs_code_string=task_obs_code_string, task_description=task_description)

    # 반복적 개선을 위한 대화 기록 초기화
    messages = [{"role": "system", "content": initial_system}, {"role": "user", "content": initial_user}]

    # 이 실험을 위해 접미사를 포함하도록 태스크 이름 수정
    task_code_string = task_code_string.replace(task, task+suffix)

    # 새로운 태스크 변형을 위한 YAML 구성 파일 생성
    create_task(ISAAC_ROOT_DIR, cfg.env.task, cfg.env.env_name, suffix)

    # ================================================================================
    # 반복적 개선을 위한 추적 변수 초기화
    # ================================================================================

    DUMMY_FAILURE = -10000.                          # 실패한 실행을 위한 센티넬 값

    # 반복에 걸친 성능 추적
    max_successes = []                                # 반복당 최고 성공률
    max_successes_reward_correlation = []             # 반복당 실제 보상과의 상관관계
    execute_rates = []                                # 성공적으로 실행된 생성 코드의 비율
    best_code_paths = []                              # 반복당 최고 성능 코드의 파일 경로

    # 전체 최고 성능 추적
    max_success_overall = DUMMY_FAILURE               # 모든 반복에서 최고 성공률
    max_success_reward_correlation_overall = DUMMY_FAILURE  # 모든 반복에서 최고 상관관계
    max_reward_code_path = None                       # 전체 최고 보상 함수 경로

    # ================================================================================
    # 2단계: 반복적 보상 함수 생성 및 개선
    # ================================================================================

    for iter in range(cfg.iteration):
        """
        Eureka 알고리즘의 메인 반복 루프. 각 반복에서:
        1. LLM을 사용하여 여러 보상 함수 후보 생성
        2. 각 후보에 대해 RL 훈련 실행
        3. 성능 평가 및 최고 후보 선택
        4. 다음 반복을 위해 LLM에 피드백 제공
        """

        # --------------------------------------------------------------------------------
        # 2.1단계: LLM을 사용한 보상 함수 후보 생성
        # --------------------------------------------------------------------------------

        responses = []                    # 이 반복을 위한 모든 LLM 응답 저장
        response_cur = None              # API로부터의 현재 응답
        total_samples = 0                # 생성한 샘플 수 추적
        total_token = 0                  # 사용된 총 토큰 추적 (비용 모니터링용)
        total_completion_token = 0       # 완료 토큰 특별 추적

        # 적응적 배치 크기: GPT-3.5는 더 큰 배치를 처리할 수 있지만, GPT-4는 제한적
        chunk_size = cfg.sample if "gpt-3.5" in model else 4

        logging.info(f"Iteration {iter}: Generating {cfg.sample} samples with {cfg.model}")

        # 요청된 수의 보상 함수 샘플 생성
        while True:
            if total_samples >= cfg.sample:
                break

            # API 실패에 대한 지수 백오프를 포함한 재시도 로직
            for attempt in range(1000):
                try:
                    # OpenAI API를 호출하여 보상 함수 후보 생성
                    response_cur = openai.ChatCompletion.create(
                        model=model,
                        messages=messages,              # 반복적 피드백이 포함된 대화 기록
                        temperature=cfg.temperature,   # 무작위성/창의성 제어
                        n=chunk_size                   # 이 호출에서 생성할 후보 수
                    )
                    total_samples += chunk_size

                    break
                except Exception as e:
                    # 많이 실패했다면, 속도 제한을 피하기 위해 배치 크기 감소
                    if attempt >= 10:
                        chunk_size = max(int(chunk_size / 2), 1)
                        print("Current Chunk Size", chunk_size)
                    logging.info(f"Attempt {attempt+1} failed with error: {e}")
                    time.sleep(1)  # 재시도 전 잠시 대기

            # 응답을 완전히 받지 못하면 중단
            if response_cur is None:
                logging.info("Code terminated due to too many failed attempts!")
                exit()

            # 응답 누적 및 토큰 사용량 추적
            responses.extend(response_cur["choices"])
            prompt_tokens = response_cur["usage"]["prompt_tokens"]
            total_completion_token += response_cur["usage"]["completion_tokens"]
            total_token += response_cur["usage"]["total_tokens"]

        # 하나의 샘플만 생성하는 경우 LLM 출력 로그 (디버깅용)
        if cfg.sample == 1:
            logging.info(f"Iteration {iter}: GPT Output:\n " + responses[0]["message"]["content"] + "\n")

        # 비용 추적 및 디버깅을 위한 토큰 사용량 로그
        logging.info(f"Iteration {iter}: Prompt Tokens: {prompt_tokens}, Completion Tokens: {total_completion_token}, Total Tokens: {total_token}")

        # --------------------------------------------------------------------------------
        # 2.2단계: 생성된 각 보상 함수 처리
        # --------------------------------------------------------------------------------

        code_runs = []   # 추출된 보상 함수 코드 저장
        rl_runs = []     # RL 훈련 프로세스 핸들 저장

        for response_id in range(cfg.sample):
            response_cur = responses[response_id]["message"]["content"]
            logging.info(f"Iteration {iter}: Processing Code Run {response_id}")

            # 여러 정규식 패턴을 사용하여 LLM 응답에서 Python 코드 추출
            # LLM은 다양한 방식으로 코드를 형식화할 수 있으므로, 여러 추출 방법을 시도
            patterns = [
                r'```python(.*?)```',    # 표준 Python 코드 블록
                r'```(.*?)```',          # 일반 코드 블록
                r'"""(.*?)"""',          # 삼중 따옴표 문자열
                r'""(.*?)""',            # 이중 따옴표 문자열
                r'"(.*?)"',              # 단일 따옴표 문자열
            ]

            code_string = None
            for pattern in patterns:
                code_match = re.search(pattern, response_cur, re.DOTALL)
                if code_match is not None:
                    code_string = code_match.group(1).strip()
                    break

            # 코드 블록을 찾지 못하면 전체 응답 사용
            code_string = response_cur if not code_string else code_string

            # 불필요한 임포트와 전문을 제거하여 추출된 코드 정리
            # 보상 함수 정의 자체만 원함
            lines = code_string.split("\n")
            for i, line in enumerate(lines):
                if line.strip().startswith("def "):
                    code_string = "\n".join(lines[i:])
                    break

            # 함수 시그니처가 유효한지 확인하기 위해 파싱
            # 통합을 위한 함수 이름과 매개변수도 추출
            try:
                gpt_reward_signature, input_lst = get_function_signature(code_string)
            except Exception as e:
                logging.info(f"Iteration {iter}: Code Run {response_id} cannot parse function signature!")
                continue

            # 유효한 보상 함수 코드 저장
            code_runs.append(code_string)

            # --------------------------------------------------------------------------------
            # 2.3단계: 보상 함수를 환경 코드에 통합
            # --------------------------------------------------------------------------------

            # 생성된 함수를 호출하는 보상 통합 코드 생성
            # GPT 생성 보상 함수를 환경의 보상 계산에 통합
            reward_signature = [
                f"self.rew_buf[:], self.rew_dict = {gpt_reward_signature}",          # 생성된 함수 호출
                f"self.extras['gpt_reward'] = self.rew_buf.mean()",                 # 로깅을 위한 평균 보상 저장
                f"for rew_state in self.rew_dict: self.extras[rew_state] = self.rew_dict[rew_state].mean()",  # 보상 구성 요소 로그
            ]

            # Python 코드를 위한 적절한 들여쓰기 추가
            indent = " " * 8
            reward_signature = "\n".join([indent + line for line in reward_signature])

            # 환경의 compute_reward 메서드에 보상 계산 삽입
            # 다른 환경은 다른 함수 시그니처를 가질 수 있음
            if "def compute_reward(self)" in task_code_string:
                task_code_string_iter = task_code_string.replace("def compute_reward(self):", "def compute_reward(self):\n" + reward_signature)
            elif "def compute_reward(self, actions)" in task_code_string:
                task_code_string_iter = task_code_string.replace("def compute_reward(self, actions):", "def compute_reward(self, actions):\n" + reward_signature)
            else:
                raise NotImplementedError

            # --------------------------------------------------------------------------------
            # 2.4단계: 생성된 환경 코드 저장 및 훈련 준비
            # --------------------------------------------------------------------------------

            # 통합된 보상 함수와 함께 완전한 환경 파일 작성
            with open(output_file, 'w') as file:
                file.writelines(task_code_string_iter + '\n')  # 보상 통합이 포함된 기본 환경 코드

                # 보상 함수를 위한 필요한 임포트 추가
                file.writelines("from typing import Tuple, Dict" + '\n')
                file.writelines("import math" + '\n')
                file.writelines("import torch" + '\n')
                file.writelines("from torch import Tensor" + '\n')

                # 성능을 위해 JIT 컴파일 데코레이터 추가 (없는 경우)
                if "@torch.jit.script" not in code_string:
                    code_string = "@torch.jit.script\n" + code_string
                file.writelines(code_string + '\n')

            # 보상 함수만 저장 (디버깅 및 분석용)
            with open(f"env_iter{iter}_response{response_id}_rewardonly.py", 'w') as file:
                file.writelines(code_string + '\n')

            # 기록 보관을 위한 완전한 환경 파일 복사
            shutil.copy(output_file, f"env_iter{iter}_response{response_id}.py")

            # --------------------------------------------------------------------------------
            # 2.5단계: 생성된 보상 함수로 RL 훈련 실행
            # --------------------------------------------------------------------------------

            # 훈련을 위해 가장 많은 사용 가능한 메모리를 가진 GPU 찾기
            set_freest_gpu()

            # 생성된 보상 함수로 RL 훈련 프로세스 시작
            rl_filepath = f"env_iter{iter}_response{response_id}.txt"
            with open(rl_filepath, 'w') as f:
                # 적절한 플래그로 훈련 프로세스 시작
                process = subprocess.Popen(['python', '-u', f'{ISAAC_ROOT_DIR}/train.py',
                                            'hydra/output=subprocess',           # 서브프로세스 출력 모드 사용
                                            f'task={task}{suffix}',              # 태스크 변형 지정
                                            f'wandb_activate={cfg.use_wandb}',   # Weights & Biases 로깅
                                            f'wandb_entity={cfg.wandb_username}',
                                            f'wandb_project={cfg.wandb_project}',
                                            f'headless={not cfg.capture_video}', # 효율성을 위한 헤드리스 모드
                                            f'capture_video={cfg.capture_video}',
                                            'force_render=False',
                                            f'max_iterations={cfg.max_iterations}'],
                                            stdout=f, stderr=f)

            # 진행하기 전에 훈련 완료 대기
            block_until_training(rl_filepath, log_status=True, iter_num=iter, response_id=response_id)
            rl_runs.append(process)

        # --------------------------------------------------------------------------------
        # 2.6단계: 훈련 결과 평가 및 피드백 구성
        # --------------------------------------------------------------------------------

        # 평가 결과를 위한 컨테이너 초기화
        code_feedbacks = []      # 각 코드 샘플에 대한 피드백 메시지
        contents = []            # 각 샘플에 대한 완전한 피드백 내용
        successes = []           # 각 샘플의 성공률
        reward_correlations = [] # 실제 보상과의 상관관계
        code_paths = []          # 생성된 각 환경의 파일 경로

        exec_success = False     # 실행 성공 여부를 추적하는 플래그

        # 각 RL 훈련 실행의 결과 처리
        for response_id, (code_run, rl_run) in enumerate(zip(code_runs, rl_runs)):
            rl_run.communicate()  # 프로세스 완료 대기
            rl_filepath = f"env_iter{iter}_response{response_id}.txt"
            code_paths.append(f"env_iter{iter}_response{response_id}.py")

            # 훈련 출력 읽기
            try:
                with open(rl_filepath, 'r') as f:
                    stdout_str = f.read()
            except:
                # 훈련 출력 파일을 읽을 수 없는 경우 처리
                content = execution_error_feedback.format(traceback_msg="Code Run cannot be executed due to function signature error! Please re-write an entirely new reward function!")
                content += code_output_tip
                contents.append(content)
                successes.append(DUMMY_FAILURE)
                reward_correlations.append(DUMMY_FAILURE)
                continue

            content = ''
            # 훈련 출력에서 오류 메시지 추출
            traceback_msg = filter_traceback(stdout_str)

            if traceback_msg == '':
                # --------------------------------------------------------------------------------
                # 훈련 성공 - 성능 분석 및 피드백 제공
                # --------------------------------------------------------------------------------
                exec_success = True

                # 훈련 출력에서 텐서보드 로그 디렉토리 추출
                lines = stdout_str.split('\n')
                for i, line in enumerate(lines):
                    if line.startswith('Tensorboard Directory:'):
                        break
                tensorboard_logdir = line.split(':')[-1].strip()

                # 훈련 메트릭 로드 및 분석
                tensorboard_logs = load_tensorboard_logs(tensorboard_logdir)
                max_iterations = np.array(tensorboard_logs['gt_reward']).shape[0]
                epoch_freq = max(int(max_iterations // 10), 1)  # 피드백을 위해 10개 데이터 포인트 샘플링

                content += policy_feedback.format(epoch_freq=epoch_freq)

                # 인간이 설계한 보상과 GPT가 생성한 보상 간의 상관관계 계산
                # 생성된 보상이 전문가 지식과 얼마나 잘 일치하는지 측정
                if "gt_reward" in tensorboard_logs and "gpt_reward" in tensorboard_logs:
                    gt_reward = np.array(tensorboard_logs["gt_reward"])
                    gpt_reward = np.array(tensorboard_logs["gpt_reward"])
                    reward_correlation = np.corrcoef(gt_reward, gpt_reward)[0, 1]
                    reward_correlations.append(reward_correlation)

                # 훈련 성능에 대한 상세한 피드백 구성
                for metric in tensorboard_logs:
                    if "/" not in metric:  # 계층적 메트릭 건너뛰기
                        # 간결한 피드백을 위해 일정한 간격으로 메트릭 샘플링
                        metric_cur = ['{:.2f}'.format(x) for x in tensorboard_logs[metric][::epoch_freq]]
                        metric_cur_max = max(tensorboard_logs[metric])
                        metric_cur_mean = sum(tensorboard_logs[metric]) / len(tensorboard_logs[metric])

                        # 성공률 추출 (기본 최적화 목표)
                        if "consecutive_successes" == metric:
                            successes.append(metric_cur_max)

                        metric_cur_min = min(tensorboard_logs[metric])

                        # 비보상 메트릭에 대한 피드백 형식화
                        if metric != "gt_reward" and metric != "gpt_reward":
                            if metric != "consecutive_successes":
                                metric_name = metric
                            else:
                                metric_name = "task_score"
                            content += f"{metric_name}: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"
                        else:
                            # 성공률을 사용할 수 없을 때 실제 점수 제공
                            if "consecutive_successes" not in tensorboard_logs:
                                content += f"ground-truth score: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"

                code_feedbacks.append(code_feedback)
                content += code_feedback
            else:
                # --------------------------------------------------------------------------------
                # 훈련 실패 - 오류 피드백 제공
                # --------------------------------------------------------------------------------
                successes.append(DUMMY_FAILURE)
                reward_correlations.append(DUMMY_FAILURE)
                content += execution_error_feedback.format(traceback_msg=traceback_msg)

            content += code_output_tip
            contents.append(content)

        # --------------------------------------------------------------------------------
        # 2.7단계: 완전한 실패 처리 및 최고 후보 선택
        # --------------------------------------------------------------------------------

        # 모든 코드 생성 시도가 실패한 경우, 반복 재시도
        if not exec_success and cfg.sample != 1:
            execute_rates.append(0.)
            max_successes.append(DUMMY_FAILURE)
            max_successes_reward_correlation.append(DUMMY_FAILURE)
            best_code_paths.append(None)
            logging.info("All code generation failed! Repeat this iteration from the current message checkpoint!")
            continue

        # 성공률을 기반으로 최고 성능 보상 함수 선택
        best_sample_idx = np.argmax(np.array(successes))
        best_content = contents[best_sample_idx]

        max_success = successes[best_sample_idx]
        max_success_reward_correlation = reward_correlations[best_sample_idx]
        execute_rate = np.sum(np.array(successes) >= 0.) / cfg.sample  # 성공적으로 실행된 비율

        # 이번 반복에서 더 나은 솔루션을 발견했다면 전역 최고값 업데이트
        if max_success > max_success_overall:
            max_success_overall = max_success
            max_success_reward_correlation_overall = max_success_reward_correlation
            max_reward_code_path = code_paths[best_sample_idx]

        # 이번 반복의 성능 메트릭 기록
        execute_rates.append(execute_rate)
        max_successes.append(max_success)
        max_successes_reward_correlation.append(max_success_reward_correlation)
        best_code_paths.append(code_paths[best_sample_idx])

        # 반복 결과 로그
        logging.info(f"Iteration {iter}: Max Success: {max_success}, Execute Rate: {execute_rate}, Max Success Reward Correlation: {max_success_reward_correlation}")
        logging.info(f"Iteration {iter}: Best Generation ID: {best_sample_idx}")
        logging.info(f"Iteration {iter}: GPT Output Content:\n" +  responses[best_sample_idx]["message"]["content"] + "\n")
        logging.info(f"Iteration {iter}: User Content:\n" + best_content + "\n")

        # --------------------------------------------------------------------------------
        # 2.8단계: 진행상황 시각화 및 결과 저장
        # --------------------------------------------------------------------------------

        # 반복에 따른 성공률과 실행률을 보여주는 진행상황 그래프 생성
        fig, axs = plt.subplots(2, figsize=(6, 6))
        fig.suptitle(f'{cfg.env.task}')

        x_axis = np.arange(len(max_successes))

        axs[0].plot(x_axis, np.array(max_successes))
        axs[0].set_title("Max Success")
        axs[0].set_xlabel("Iteration")

        axs[1].plot(x_axis, np.array(execute_rates))
        axs[1].set_title("Execute Rate")
        axs[1].set_xlabel("Iteration")

        fig.tight_layout(pad=3.0)
        plt.savefig('summary.png')

        # 정량적 결과 저장
        np.savez('summary.npz', max_successes=max_successes, execute_rates=execute_rates,
                best_code_paths=best_code_paths, max_successes_reward_correlation=max_successes_reward_correlation)

        # --------------------------------------------------------------------------------
        # 2.9단계: 다음 반복을 위한 대화 기록 업데이트
        # --------------------------------------------------------------------------------

        # 최고 응답과 피드백을 포함하도록 LLM과의 대화 업데이트
        # 이전 시도의 컨텍스트를 제공하여 반복적 개선 가능
        if len(messages) == 2:
            # 첫 번째 반복: 어시스턴트 응답과 사용자 피드백 추가
            messages += [{"role": "assistant", "content": responses[best_sample_idx]["message"]["content"]}]
            messages += [{"role": "user", "content": best_content}]
        else:
            # 후속 반복: 마지막 어시스턴트 응답과 사용자 피드백 업데이트
            assert len(messages) == 4
            messages[-2] = {"role": "assistant", "content": responses[best_sample_idx]["message"]["content"]}
            messages[-1] = {"role": "user", "content": best_content}

        # 분석 및 디버깅을 위한 대화 기록 저장
        with open('messages.json', 'w') as file:
            json.dump(messages, file, indent=4)

    # ================================================================================
    # 3단계: 최고 보상 함수의 최종 평가
    # ================================================================================

    # 작동하는 보상 함수를 찾았는지 확인
    if max_reward_code_path is None:
        logging.info("All iterations of code generation failed, aborting...")
        logging.info("Please double check the output env_iter*_response*.txt files for repeating errors!")
        exit()

    # 최종 결과 로그
    logging.info(f"Task: {task}, Max Training Success {max_success_overall}, Correlation {max_success_reward_correlation_overall}, Best Reward Code Path: {max_reward_code_path}")
    logging.info(f"Evaluating best reward code {cfg.num_eval} times")

    # 최고 보상 함수를 최종 출력 위치로 복사
    shutil.copy(max_reward_code_path, output_file)

    # --------------------------------------------------------------------------------
    # 통계적 유의성을 위한 다중 평가 실행
    # --------------------------------------------------------------------------------

    eval_runs = []
    for i in range(cfg.num_eval):
        set_freest_gpu()

        # 강건한 평가를 위해 다른 랜덤 시드로 RL 훈련 실행
        rl_filepath = f"reward_code_eval{i}.txt"
        with open(rl_filepath, 'w') as f:
            process = subprocess.Popen(['python', '-u', f'{ISAAC_ROOT_DIR}/train.py',
                                        'hydra/output=subprocess',
                                        f'task={task}{suffix}',
                                        f'wandb_activate={cfg.use_wandb}',
                                        f'wandb_entity={cfg.wandb_username}',
                                        f'wandb_project={cfg.wandb_project}',
                                        f'headless={not cfg.capture_video}',
                                        f'capture_video={cfg.capture_video}',
                                        'force_render=False',
                                        f'seed={i}',  # 각 평가마다 다른 시드
                                        ],
                                        stdout=f, stderr=f)

        block_until_training(rl_filepath)
        eval_runs.append(process)

    # --------------------------------------------------------------------------------
    # 최종 평가 결과 분석
    # --------------------------------------------------------------------------------

    reward_code_final_successes = []      # 모든 평가 실행에서의 성공률
    reward_code_correlations_final = []   # 모든 평가 실행에서의 보상 상관관계

    for i, rl_run in enumerate(eval_runs):
        rl_run.communicate()  # 완료 대기
        rl_filepath = f"reward_code_eval{i}.txt"

        # 훈련 출력 읽기
        with open(rl_filepath, 'r') as f:
            stdout_str = f.read()

        # 텐서보드 디렉토리 추출
        lines = stdout_str.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('Tensorboard Directory:'):
                break
        tensorboard_logdir = line.split(':')[-1].strip()

        # 메트릭 로드 및 성공률 추출
        tensorboard_logs = load_tensorboard_logs(tensorboard_logdir)
        max_success = max(tensorboard_logs['consecutive_successes'])
        reward_code_final_successes.append(max_success)

        # 사용 가능한 경우 보상 상관관계 계산
        if "gt_reward" in tensorboard_logs and "gpt_reward" in tensorboard_logs:
            gt_reward = np.array(tensorboard_logs["gt_reward"])
            gpt_reward = np.array(tensorboard_logs["gpt_reward"])
            reward_correlation = np.corrcoef(gt_reward, gpt_reward)[0, 1]
            reward_code_correlations_final.append(reward_correlation)

    # --------------------------------------------------------------------------------
    # 통계적 요약과 함께 최종 결과 보고
    # --------------------------------------------------------------------------------

    # 포괄적인 최종 평가 결과 로그
    logging.info(f"Final Success Mean: {np.mean(reward_code_final_successes)}, Std: {np.std(reward_code_final_successes)}, Raw: {reward_code_final_successes}")
    logging.info(f"Final Correlation Mean: {np.mean(reward_code_correlations_final)}, Std: {np.std(reward_code_correlations_final)}, Raw: {reward_code_correlations_final}")

    # 분석을 위한 최종 평가 결과 저장
    np.savez('final_eval.npz',
             reward_code_final_successes=reward_code_final_successes,
             reward_code_correlations_final=reward_code_correlations_final)


if __name__ == "__main__":
    main()
