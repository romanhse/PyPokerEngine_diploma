import os
import requests
from uuid import uuid4


class DeepSeekChat:
    def __init__(self, api_key, system_prompt=None):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self.sessions = {}
        self.current_session = None
        self.default_system_prompt = system_prompt or "You are a helpful assistant"

    def create_session(self, system_prompt=None):
        session_id = str(uuid4())
        self.sessions[session_id] = {
            "system_prompt": system_prompt or self.default_system_prompt,
            "messages": []
        }
        if not self.current_session:
            self.current_session = session_id
        return session_id

    def switch_session(self, session_id):
        if session_id in self.sessions:
            self.current_session = session_id
            return True
        return False

    def get_response(self, prompt, session_id=None):
        if not session_id:
            session_id = self.current_session
        if not session_id or session_id not in self.sessions:
            raise ValueError("Invalid session ID")

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
            return ai_response
        else:
            raise Exception(f"API Error: {response_data}")

    def set_system_prompt(self, new_prompt, session_id=None):
        if not session_id:
            session_id = self.current_session
        if session_id in self.sessions:
            self.sessions[session_id]["system_prompt"] = new_prompt
            return True
        return False

    def list_sessions(self):
        return list(self.sessions.keys())


if __name__ == "__main__":
    # api_key = os.getenv("DEEPSEEK_API_KEY")
    api_key = ''
    if not api_key:
        raise ValueError("Please set DEEPSEEK_API_KEY environment variable")

    chat = DeepSeekChat(api_key)

    # Создаем первый сеанс
    main_session = chat.create_session()
    print(f"Создан новый сеанс: {main_session}")

    while True:
        try:
            user_input = input("\nВы: ")

            if user_input.startswith("/"):
                command = user_input[1:].split()

                if command[0] == "new":
                    new_session = chat.create_session()
                    chat.switch_session(new_session)
                    print(f"Создан новый сеанс: {new_session}")

                elif command[0] == "switch":
                    if len(command) > 1:
                        session_id = command[1]
                        if chat.switch_session(session_id):
                            print(f"Переключен на сеанс: {session_id}")
                        else:
                            print("Неверный ID сеанса")

                elif command[0] == "system":
                    if len(command) > 1:
                        new_prompt = " ".join(command[1:])
                        chat.set_system_prompt(new_prompt)
                        print("Системный промпт обновлен")

                elif command[0] == "list":
                    sessions = chat.list_sessions()
                    print("Активные сеансы:")
                    for session in sessions:
                        print(f"- {session}")

                elif command[0] == "exit":
                    break

                else:
                    print("Неизвестная команда")

                continue

            # Получаем ответ от ИИ
            response = chat.get_response(user_input)
            print(f"\nAI: {response}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Ошибка: {str(e)}")