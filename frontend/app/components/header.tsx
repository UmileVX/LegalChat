import Image from "next/image";

import { LOGO_REDIRECT_URL, LOGO_HEIGHT, LOGO_WIDTH, } from "../constants/logo";

export default function Header() {
  return (
    <div className="z-10 max-w-5xl w-full items-center justify-between font-mono text-sm lg:flex">
      <div className="fixed bottom-0 left-0 flex h-48 w-full items-end justify-center bg-gradient-to-t from-white via-white dark:from-black dark:via-black lg:static lg:h-auto lg:w-auto lg:bg-none">
        <a
          href={LOGO_REDIRECT_URL}
          className="flex items-center justify-center font-nunito text-lg font-bold gap-2 mx-auto"
        >
          <Image
            className="rounded-xl"
            src="/logo.png"
            alt="Service Logo"
            width={LOGO_WIDTH}
            height={LOGO_HEIGHT}
            priority
          />
        </a>
      </div>
    </div>
  );
}
