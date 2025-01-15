"use client";

import Image from 'next/image'

import '../../../ai_chat_box.css';

export interface ChatInputProps {
  /** The current value of the input */
  input?: string;
  /** An input/textarea-ready onChange handler to control the value of the input */
  handleInputChange?: (
    e:
      | React.ChangeEvent<HTMLInputElement>
      | React.ChangeEvent<HTMLTextAreaElement>,
  ) => void;
  /** Form submission handler to automatically reset input and append a user message  */
  handleSubmit: (e: React.FormEvent<HTMLFormElement>) => void;
  isLoading: boolean;
  multiModal?: boolean;
  disabled?: boolean;
}


export default function ChatInput(props: ChatInputProps) {
  return (
    <>
      <form
        onSubmit={props.handleSubmit}
        className="flex items-start justify-between w-full p-4 bg-white rounded-xl shadow-xl gap-4 ai_chat_box"
      >
        <input
          autoFocus
          name="message"
          placeholder="메세지를 입력해주세요."
          className="w-full p-4 rounded-xl shadow-inner flex-1 focus:outline-none focus:ring-1 focus:ring-green-500 "
          value={props.input}
          onChange={props.handleInputChange}
        />
        <button
          disabled={props.isLoading || props.disabled}
          type="submit"
          className="p-4 text-white rounded-xl shadow-xl bg-gradient-to-r from-green-500 to-green-600 disabled:opacity-50 disabled:cursor-not-allowed w-14 h-14 p-5"
        >
          <Image
            src="/icons/send_all.png"
            width={18}
            height={18}
            alt="전송하기"
          />
        </button>
      </form>
    </>
  );
}
