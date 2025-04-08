import os
from dotenv import load_dotenv
import requests
from uuid import uuid4
import datetime

load_dotenv()

class DeepSeekChat:
    def __init__(self, api_key, system_prompt):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self.sessions = {}
        self.current_session = None
        self.default_system_prompt = system_prompt

    def create_session(self, system_prompt=None):
        session_id = str(uuid4())
        self.sessions[session_id] = {
            "system_prompt": system_prompt or self.default_system_prompt,
            "messages": []
        }
        if not self.current_session:
            self.current_session = session_id
        return session_id

    def get_response(self, prompt, session_id=None):
        if not session_id:
            session_id = self.current_session
        if not session_id or session_id not in self.sessions:
            raise ValueError("Invalid session ID")

        with open(f'{self.current_session}.txt', 'a')  as file:
            file.write(prompt +'\n')

        session = self.sessions[session_id]
        messages = [{
            "role": "system",
            "content": session["system_prompt"]
        }]
        messages += session["messages"]
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "deepseek-chat",
            "messages": messages,
            "temperature": 0.7
        }

        response = requests.post(self.base_url, headers=headers, json=data)
        response_data = response.json()

        if response.status_code == 200:
            ai_response = response_data["choices"][0]["message"]["content"]
            session["messages"].extend([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": ai_response}
            ])

            with open(f'{self.current_session}.txt', 'a') as file:
                file.write(ai_response + '\n')

            return ai_response
        else:
            raise Exception(f"API Error: {response_data}")


SYSTEM_PROMT = """
You are a professional poker player. Firstly, you receive the rules of the game. 
Then, for each round you receive all available information, based on which you have to make decision
You can either fold - then you return "f", call - then you return "c", or raise - then you return "r amount", like "r 50"
YOUR RESPONSE ALWAYS HAS TO BE ONLY OF FORM "f", "c" OR "r __amount__", NO OTHER SYMBOLS ARE AVAILABLE
DO NOT FOLD IF YOU CAN CALL FOR FREE!!!
"""
api_key = os.getenv('API_KEY')


import pypokerengine.utils.visualize_utils as U
from pypokerengine.players import BasePokerPlayer

class DeepseekPlayer(BasePokerPlayer):

  def __init__(self, input_receiver=None):
    #self.input_receiver = input_receiver if input_receiver else self.__gen_raw_input_wrapper()
    self.chat = DeepSeekChat(api_key, SYSTEM_PROMT)
    main_session = self.chat.create_session()
    print(f"Создан новый сеанс: {main_session}")

  def declare_action(self, valid_actions, hole_card, round_state):
    # print(U.visualize_declare_action(valid_actions, hole_card, round_state, self.uuid))
    for i in round_state['seats']:
        if i['name'] == 'deepseek_player':
            available_amount = int(i['stack'])
    del round_state['action_histories']
    for j in round_state['seats']:
        del j['name']
    promt = f"""
Valid actions: {valid_actions}
Your cards: {hole_card}
Your stack: {available_amount}
Game info: {round_state}
Return your action
"""
    response = self.chat.get_response(promt)
    print(f'Deepseek response {response}')
    action, amount = self.__receive_action_from_deepseek(valid_actions, response, promt)
    return action, amount

  def receive_game_start_message(self, game_info):
    promt = f"""
Game rules: {game_info}
return '+' if you understand
"""
    response = self.chat.get_response(promt)
    print(f'Deepseek response to game start {response}')
    # print(U.visualize_game_start(game_info, self.uuid))
    # self.__wait_until_input()
  def receive_round_start_message(self, round_count, hole_card, seats):
      promt = f"""
    Round count: {round_count}
    Your cards: {hole_card}
    Seats: {seats}
    return '+' if you understand
    """
      response = self.chat.get_response(promt)
      print(f'Deepseek response to round start {response}')

  def receive_street_start_message(self, street, round_state):
      promt = f"""
        Street: {street}
        Round state: {round_state}
        return '+' if you understand
        """
      response = self.chat.get_response(promt)
      print(f'Deepseek response to street start {response}')

  def receive_game_update_message(self, new_action, round_state):
    pass
    promt = f"""
        New action: {new_action}
        Round state: {round_state}
        return '+' if you understand
        """
    response = self.chat.get_response(promt)
    print(f'Deepseek response to round update {response}')

  def receive_round_result_message(self, winners, hand_info, round_state):
      promt = f"""
          Winners: {winners}
          Hand info: {hand_info}
          Round state: {round_state}
          return '+' if you understand
          """
      response = self.chat.get_response(promt)
      print(f'Deepseek response to round results {response}')

  def __wait_until_input(self):
    input("Enter some key to continue ...")

  def __gen_raw_input_wrapper(self):
    return lambda msg: input(msg)

  def __receive_action_from_deepseek(self, valid_actions, response, promt):
    if response in self.__gen_valid_flg(valid_actions) or response[0] in self.__gen_valid_flg(valid_actions):
      if response == 'f':
        return valid_actions[0]['action'], valid_actions[0]['amount']
      elif response == 'c':
        return valid_actions[1]['action'], valid_actions[1]['amount']
      elif response[0] == 'r':
        print('bebra)')
        valid_amounts = valid_actions[2]['amount']
        try:
            deepseek_amt = int(response.split()[1])
        except:
            print(f'Invalid raise input {response}')
        raise_amount = self.__receive_raise_amount_from_deepseek(valid_amounts['min'], valid_amounts['max'], deepseek_amt)
        return valid_actions[2]['action'], raise_amount
    else:
        response = self.chat.get_response(promt)
        return self.__receive_action_from_deepseek(valid_actions, response, promt)

  def __gen_valid_flg(self, valid_actions):
    flgs = ['f', 'c']
    is_raise_possible = valid_actions[2]['amount']['min'] != -1
    if is_raise_possible:
      flgs.append('r')
    return flgs

  def __receive_raise_amount_from_deepseek(self, min_amount, max_amount, deepseek_amount):
    try:
      if min_amount <= deepseek_amount and deepseek_amount <= max_amount:
        return deepseek_amount
      else:
        print("Invalid raise amount %d. Try again.")
        return self.__receive_raise_amount_from_deepseek(min_amount, max_amount, deepseek_amount)
    except:
      print("Invalid input received. Try again.")
      return self.__receive_raise_amount_from_deepseek(min_amount, max_amount, deepseek_amount)

