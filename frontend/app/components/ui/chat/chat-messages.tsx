"use client";

import { useEffect, useRef } from "react";
import ChatItem from "./chat-item";

import '../../../ai_chat_box.css';

export interface Message {
  id: string;
  content: string;
  role: string;
}

export default function ChatMessages({
  messages,
  isLoading,
  reload,
  stop,
}: {
  messages: Message[];
  isLoading?: boolean;
  stop?: () => void;
  reload?: () => void;
}) {
  const scrollableChatContainerRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    if (scrollableChatContainerRef.current) {
      scrollableChatContainerRef.current.scrollTop =
        scrollableChatContainerRef.current.scrollHeight;
    }
  };

  // messages 배열의 길이가 변할 때 스크롤 이동
  useEffect(() => {
    scrollToBottom();
  }, [messages.length]);

  // isLoading 상태가 false가 될 때 스크롤 이동
  useEffect(() => {
    if (!isLoading) {
      scrollToBottom();
    }
  }, [isLoading]);

  return (
    <div className="w-full max-w-6xl px-10 py-8 bg-white rounded-xl shadow-xl ai_chat_box">
      <div
        className="flex flex-col gap-5 divide-y h-[50vh] overflow-auto"
        ref={scrollableChatContainerRef}
      >
        {messages.map((m: Message) => (
          <ChatItem key={m.id} {...m} />
        ))}
      </div>
    </div>
  );
}
