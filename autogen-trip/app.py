# run_agents.py
import argparse
import asyncio
import getpass
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelFamily  # model_infoでfamilyを指定するのに使用

def make_model_client(provider: str):
    if provider == "openrouter":
        # ★ APIキーは環境変数 or 実行時入力で安全に扱う
        api_key = os.environ.get("OPENROUTER_API_KEY") or getpass.getpass("Enter your OpenRouter API Key: ")

        return OpenAIChatCompletionClient(
            # ★ 無料枠モデルの例（時期により変更あり）
            model="deepseek/deepseek-r1",
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",

            # ★ OpenAI互換APIでOpenAI“以外”のモデル名を使う場合は model_info が必須
            model_info={
                "vision": False,            # 画像入力が不要なら False
                "function_calling": False,  # ツール呼び出しを使わないなら False
                "json_output": False,       # JSONモード前提でなければ False
                "structured_output": False, # 構造化出力を使わないなら False
                # family は任意だが、例として R1 系列を設定（厳密一致でなくてもOK）
                "family": ModelFamily.R1,
            },

            # （任意だが推奨）OpenRouterのポリシー向けヘッダ
            default_headers={
                "HTTP-Referer": "http://localhost",   # あなたのサイト/アプリ名など
                "X-Title": "autogen-trip-sample",
            },

            # 一部プロバイダで name フィールド未対応のときは False にする
            # include_name_in_message=False,
        )
    else:
        raise ValueError("Unknown provider: " + provider)

def build_team(model_client):
    planner_agent = AssistantAgent(
        "planner_agent",
        model_client=model_client,
        description="旅行の計画を立ててくれる便利なアシスタント",
        system_message="あなたは、ユーザーのリクエストに基づいて旅行プランを提案できる便利なアシスタントです。",
    )
    local_agent = AssistantAgent(
        "local_agent",
        model_client=model_client,
        description="地元のアクティビティや訪問先を提案できる地元アシスタント",
        system_message="あなたは、本物で興味深い地元のアクティビティや訪問する場所をユーザーに提案し、提供されたコンテキスト情報を利用できる便利なアシスタントです。",
    )
    language_agent = AssistantAgent(
        "language_agent",
        model_client=model_client,
        description="特定の目的地に関する言語のヒントを提供できる便利なアシスタント",
        system_message="あなたは、旅行計画を検討し、特定の目的地での公用語やコミュニケーションの課題に対処する最善の方法に関する重要なヒントについてフィードバックを提供できる便利なアシスタントです。計画に言語に関するヒントがすでに含まれている場合は、その計画が満足のいくものであることを根拠を添えて言及できます。",
    )
    travel_summary_agent = AssistantAgent(
        "travel_summary_agent",
        model_client=model_client,
        description="旅行計画をまとめるのに役立つアシスタント",
        system_message="あなたは、他のエージェントからの提案やアドバイスをすべて取り入れ、詳細な最終的な旅行計画を提供できる、役に立つアシスタントです。最終計画が統合され、完全であることを確認する必要があります。最終的な対応は完全な計画でなければなりません。計画が完了し、すべてのパースペクティブが統合されたら、TERMINATE で応答できます。",
    )
    termination = TextMentionTermination("TERMINATE")

    group_chat = RoundRobinGroupChat(
        [planner_agent, local_agent, language_agent, travel_summary_agent],
        termination_condition=termination,
        max_turns=10
    )
    return group_chat

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="ネパールへの3日間の旅行を計画してください。")
    args = parser.parse_args()

    model_client = make_model_client("openrouter")
    team = build_team(model_client)
    await Console(team.run_stream(task=args.task))

if __name__ == "__main__":
    asyncio.run(main())

