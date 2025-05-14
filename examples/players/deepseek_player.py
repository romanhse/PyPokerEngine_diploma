import os
from dotenv import load_dotenv
import requests
from uuid import uuid4
from pypokerengine.players import BasePokerPlayer
import datetime
import time
from treys import Evaluator, Card


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

        def messages_length(msgs):
            return sum(len(m["content"]) for m in msgs)

        while messages_length(messages) > 60000 and len(messages) > 2:
            # Remove the second message (first user/assistant message, preserve system + prompt)
            del messages[1]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "deepseek-chat",
            "messages": messages,
            "temperature": 0.7
        }

        for i in range(5):
            response = requests.post(self.base_url, headers=headers, json=data)
            response_data = response.json()
            if response.status_code == 200:
                break
            else:
                time.sleep(5)

        if response.status_code == 200:
            # reasoning_content = response_data["choices"][0]["message"]["reasoning_content"]
            ai_response = response_data["choices"][0]["message"]["content"]
            session["messages"].extend([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": ai_response}
            ])

            with open(f'{self.current_session}.txt', 'a') as file:
                file.write(ai_response + '\n')
                # file.write(reasoning_content + '\n')

            return ai_response
        else:
            raise Exception(f"API Error: {response_data}")


SYSTEM_PROMT = """
You are a professional No-Limit Texas Hold'em player. Make decisions based on:
- Your hole cards
- Community cards
- Pot size
- Stack sizes
- Opponent behavior

Valid actions:
- Fold (return "f")
- Call/Check (return "c")
- Raise (return "r X" where X is between [min] and [max])

DO NOT FOLD BEFORE THE FLOP!!! Consider position and pot odds. Analyze your opponent's behavior and bluff strategically.
Do not forget, that your opponent may bluff as well
Respond ONLY with "f", "c", or "r X". No explanations.
"""
SYSTEM_PROMT_NEW = """
You are a professional No-Limit Texas Hold'em player. Make decisions based on:
- Your hole cards
- Community cards
- Pot size
- Stack sizes
- Opponent behavior

Valid actions:
- Fold (return "f")
- Call/Check (return "c")
- Raise (return "r X" where X is between [min] and [max])

DO NOT FOLD BEFORE THE FLOP!!! Consider position and pot odds. Analyze your opponent's behavior and bluff strategically.
Do not forget, that your opponent may bluff as well
Starting form flop position you will receive the relative strength of your hand with regards to current community cards.
It will be in a range from 0 to 1, where 0 - the weakest hand, 1- the strongest. Take that info into account.
Respond ONLY with "f", "c", or "r X". No explanations. 
"""
api_key = os.getenv('API_KEY')




class DeepseekPlayer(BasePokerPlayer):

  def __init__(self, deepseek_log_file, other_log_file, new_version = False):
    self.deepseek_file = deepseek_log_file
    self.other_file = other_log_file
    self.new_version = new_version
    if self.new_version:
        self.chat = DeepSeekChat(api_key, SYSTEM_PROMT_NEW)
    else:
        self.chat = DeepSeekChat(api_key, SYSTEM_PROMT)
    self.our_stack = []
    self.opp_stack = []
    main_session = self.chat.create_session()
    print(f"Создан новый сеанс: {main_session}")

  def declare_action(self, valid_actions, hole_card, round_state):
    seats = self.get_seats(round_state['seats'])
    del round_state['action_histories']
    for j in round_state['seats']:
        del j['name']
    have_strength = False
    if len(round_state['community_card']) > 0:
        have_strength = True
        hand_strength = self.evaluate_hand_strength(hole_card, round_state['community_card'])
    promt = \
f"""Valid actions: {valid_actions}
Your cards: {hole_card}
Game info: 
pot:{round_state['pot']['main']['amount']}
community cards: {round_state['community_card']}
Your stack: {seats[0]}, your state: {seats[1]}
Opponent stack: {seats[2]}, opponent state: {seats[3]}
Return your action
"""
    if self.new_version:
        if have_strength:
            promt = \
                f"""Valid actions: {valid_actions}
            Your cards: {hole_card}
            Game info: 
            pot:{round_state['pot']['main']['amount']}
            community cards: {round_state['community_card']}
            Your stack: {seats[0]}, your state: {seats[1]}
            Opponent stack: {seats[2]}, opponent state: {seats[3]}
            Your sthength: {hand_strength}
            Return your action
            """

    response = self.chat.get_response(promt)
    # print(f'Deepseek response {response}')
    action, amount = self.__receive_action_from_deepseek(valid_actions, response, promt)
    if have_strength and amount > 0:
        if self.new_version:
            with open('deepseek_new_bluff.txt', 'a') as f:
                f.write(f'{hand_strength}, {action}, {amount}, \n')
        else:
            with open('deepseek_old_bluff.txt', 'a') as f:
                f.write(f'{hand_strength}, {action}, {amount}, \n')

    return action, amount

  def receive_game_start_message(self, game_info):
    promt = \
f"""Game rules: 
№ of players: {game_info['player_num']}
Initial stack: {game_info['rule']['initial_stack']}
№ of rounds: {game_info['rule']['max_round']}
Small blind: {game_info['rule']['small_blind_amount']} , big blind: {2*game_info['rule']['small_blind_amount']}
return '+' if you understand
"""
    response = self.chat.get_response(promt)

    if self.our_stack and not self.new_version:
        with open(self.deepseek_file, 'a') as f:
            for i in self.our_stack:
                f.write(i + ' ')
            f.write('\n')
        with open(self.other_file, 'a') as f:
            for i in self.opp_stack:
                f.write(i + ' ')
            f.write('\n')
        print('Files updated')
    self.opp_stack = ['100']
    self.our_stack = ['100']
    # print(f'Deepseek response to game start {response}')

  def get_seats(self, seats):
      for i in seats:
          if i['uuid'] == self.uuid:
              our_amount = int(i['stack'])
              our_state = i['state']
          else:
              opp_amount = int(i['stack'])
              opp_state = i['state']
      return [our_amount, our_state, opp_amount, opp_state]
  def receive_round_start_message(self, round_count, hole_card, seats):
      seats = self.get_seats(seats)
      promt = \
f"""Round count: {round_count}
Your cards: {hole_card}
Your stack: {seats[0]}, your state: {seats[1]}
Opponent stack: {seats[2]}, opponent state: {seats[3]}
return '+' if you understand
"""
      response = self.chat.get_response(promt)
      # print(f'Deepseek response to round start {response}')

  def receive_street_start_message(self, street, round_state):
      promt = \
f"""Street: {street}
Round state: 
pot:{round_state['pot']['main']['amount']}
community cards: {round_state['community_card']}
return '+' if you understand
"""
      response = self.chat.get_response(promt)
      # print(f'Deepseek response to street start {response}')

  def receive_game_update_message(self, new_action, round_state):
    if new_action['player_uuid'] != self.uuid:
        promt = \
f"""Your opponent declared action: {new_action['action']} with amount {new_action['amount']}
return '+' if you understand
"""
        response = self.chat.get_response(promt)
        # print(f'Deepseek response to round update {response}')

  def receive_round_result_message(self, winners, hand_info, round_state):
      seats = self.get_seats(round_state['seats'])
      self.our_stack.append(str(seats[0]))
      self.opp_stack.append(str(seats[2]))

      if winners[0]['uuid'] == self.uuid:
          winner = 'You'
      else:
          winner = 'Opponent'
      promt = \
f"""Winner: {winner}
Hand info: {hand_info}
return '+' if you understand
"""
      response = self.chat.get_response(promt)
      # print(f'Deepseek response to round results {response}')

  def __wait_until_input(self):
    input("Enter some key to continue ...")

  def __gen_raw_input_wrapper(self):
    return lambda msg: input(msg)

  def __receive_action_from_deepseek(self, valid_actions, response, promt):
      retry = False
      if response == 'f':
          return valid_actions[0]['action'], valid_actions[0]['amount']
      elif response == 'c':
          return valid_actions[1]['action'], valid_actions[1]['amount']
      elif response[0] == 'r':
          valid_amounts = valid_actions[2]['amount']
          try:
              deepseek_amt = int(response.split()[1])
              if valid_amounts['min'] <= deepseek_amt and deepseek_amt <= valid_amounts['max']:
                  return valid_actions[2]['action'], deepseek_amt
              else:
                  retry = True
          except:
              print(f'Invalid raise input {response}')
              retry = True
      else:
          retry = True
      if retry:
          print('Retry getting action from deepseek')
          response = self.chat.get_response(promt)
          return self.__receive_action_from_deepseek(valid_actions, response, promt)

  def evaluate_hand_strength(self, hole_card, community_cards):
      evaluator = Evaluator()
      hand = []
      board = []
      for card in hole_card:
          new_format = card[1] + card[0].lower()
          hand.append(Card.new(new_format))
      for card in community_cards:
          new_format = card[1] + card[0].lower()
          board.append(Card.new(new_format))
      if len(board) < 3:
          return 0.5
      else:
          score = evaluator.evaluate(hand, board)
          percentile = evaluator.get_five_card_rank_percentage(score)
          return 1 - percentile