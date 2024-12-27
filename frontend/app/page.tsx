import Header from "@/app/components/header";
import SihmHeader from "./components/sihm-header";
import ChatSection from "./components/chat-section";

export default function Home() {
  if (process.env.IS_LOCAL_BUILD === "true" || process.env.FOR_SIHM === "true") {
    return (
      <>
        <SihmHeader />
        <main className="flex min-h-fit flex-col items-center gap-10 p-24 background-gradient">
          <ChatSection />
        </main>
      </>
    )
  }

  return (
    <main className="flex min-h-screen flex-col items-center gap-10 p-24 background-gradient">
      <Header />
      <ChatSection />
    </main>
  );
}
