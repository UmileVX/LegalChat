'use client';

import '../ai_menu.css';


export default function MenuBar() {
    return (
        <div className="ai_chat_menu bg-white">
            <div className="menu_title"><span className="menu_title_name">Menu</span><span className="mdi mdi-sort-variant sort_icon"></span></div>
            <div className="subMenu_title selected"><span className="mdi mdi-robot-outline robot_icon"></span><span className="subMenu_title_name">AI 챗봇</span></div>
        </div>
    );
}
