'use client';

import Image from "next/image";

import { LOGO_REDIRECT_URL, LOGO_HEIGHT, LOGO_WIDTH, } from "../constants/sihm_logo";
import {
    doLogout,
    goToMyPage,
    goToWorkSpace,
} from "../utils/header_utils";

import '../header.css';


export default function SihmHeader() {
  return (
    <header>
        <div className="head-wrap">
            <Image
                id="sihm_logo"
                className="head-logo"
                src="/sihm_logo.png"
                alt="Service Logo"
                width={LOGO_WIDTH}
                height={LOGO_HEIGHT}
                priority
            />
            <nav className="head-menu-nav">
                <ul className="head-menu-nav-ul">
                    <li className="main-nav00"><a id="workspacelink" href="/redirect_depts">사업장 관리</a></li>
                    <li className="main-nav01"><a id="chemlink" href="/toxfree">화학물질</a></li>
                    <li className="main-nav02"><a id="musclink" href="/redirect_groups">근골격계 부담작업</a></li>
                    <li className="main-nav03 focused"><a id="chatbotlink" href="#">산업보건 챗봇</a></li>
                </ul>
            </nav>

            <div className="user_info">
                <span id="company"></span> / <span id="user"></span>
            </div>

            <div className="dropdown">
                <Image
                    className="user_profile"
                    src="/icons/user_dropdown.png"
                    alt="user icon"
                    width="20"
                    height="20"
                    priority
                />
                <div className="dropdown-content" id="dropdown-content">
                    <a href="javascript:void(0)" id="email">email</a>
                    <a href="javascript:void(0)" onClick={goToMyPage}>
                        <Image
                            className="account_icon"
                            src="/icons/user.png"
                            alt="account icon"
                            width="20"
                            height="20"
                            priority
                        />
                        마이페이지
                    </a>
                    <a href="javascript:void(0)" onClick={goToWorkSpace}>
                        <Image
                            className="workplace_icon"
                            src="/icons/workplace.png"
                            alt="workplace icon"
                            width="20"
                            height="20"
                            priority
                        />
                        사업장 목록
                    </a>
                    <a href="javascript:void(0)" onClick={doLogout}>
                        <Image
                            className="logout_icon"
                            src="/icons/logout.png"
                            alt="logout icon"
                            width="20"
                            height="20"
                            priority
                        />
                        로그아웃
                    </a>
                </div>
            </div>
        </div>
    </header>
  );
}
