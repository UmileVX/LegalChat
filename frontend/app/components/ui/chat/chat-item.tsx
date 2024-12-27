// "use client";

// import ChatAvatar from "./chat-avatar";
// import { Message } from "./chat-messages";

// export default function ChatItem(message: Message) {
//   return (
//     <div className="flex items-start gap-4 pt-5">
//       <ChatAvatar {...message} />
//       <p className="break-words whitespace-pre-wrap">{message.content}</p>
//     </div>
//   );
// }

"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import ChatAvatar from "./chat-avatar";
import { Message } from "./chat-messages";


export default function ChatItem(message: Message) {
  return (
    <div className="flex items-start gap-4 pt-5">
      <ChatAvatar {...message} />
      {/* Tailwind의 prose 클래스와 break-words 등을 함께 사용 */}
      <div className="prose break-words max-w-none">
        <ReactMarkdown 
          remarkPlugins={[remarkGfm]} 
          /* 필요한 경우 rehypePlugins, allowDangerousHtml 등 설정 가능 */
        >
          {message.content}
        </ReactMarkdown>
      </div>
    </div>
  );
}
