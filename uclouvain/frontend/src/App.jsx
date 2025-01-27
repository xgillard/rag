import { useState } from 'react';
import { AppBar, Box, Container, CssBaseline, Typography } from '@mui/material';
import StartConversation from './components/StartConversation';

function App() {
  const [text, set_text] = useState();
  return (
      <>
        <CssBaseline />
        <Container maxWidth="lg" sx={{alignContent: "center", paddingTop: "10em"}}>
            <StartConversation title="Que puis-je faire pour vous aujourd'hui ?" text={text} set_text={set_text}></StartConversation>
        </Container>
      </>
  )
}

export default App
