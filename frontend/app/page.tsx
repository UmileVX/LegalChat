import Header from "@/app/components/header";
import SihmHeader from "./components/sihm-header";
import ChatSection from "./components/chat-section";
import MenuBar from "./components/menu-bar";


export default function Home() {
  if (process.env.IS_LOCAL_BUILD === "true" || process.env.FOR_SIHM === "true") {
    return (
      <>
        <SihmHeader />
        <main className="flex h-dvh-sihm flex-col items-center gap-10">
          <div className="ai_chat_css">
            <MenuBar />
            <ChatSection />
          </div>
        </main>
      </>
    );
  }

  return (
    <main className="flex min-h-screen flex-col items-center gap-10 p-14 background-gradient">
      <Header />
      <ChatSection />
    </main>
  );
}
